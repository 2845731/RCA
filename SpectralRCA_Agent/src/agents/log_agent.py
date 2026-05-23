from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import pandas as pd

from src.agents.base_agent import BaseAgent
from src.config import SpectralRCAConfig
from src.data_loader import DataLoader
from src.llm.client import LLMClient
from src.llm.prompts import LOG_ANALYSIS_SYSTEM, format_log_prompt
from src.schemas import EvidenceReport, RCAQuery


class LogAgent(BaseAgent):
    """Log analysis agent with LLM-enhanced error pattern summarization.

    When LLM is available, it generates natural language summaries
    for each component's error patterns. When LLM is unavailable,
    it falls back to rule-based keyword matching.
    """

    def __init__(
        self,
        config: Optional[SpectralRCAConfig] = None,
        llm_client: Optional[LLMClient] = None,
    ) -> None:
        super().__init__(name="LogAgent", config=config)
        self.llm = llm_client
        self.error_keywords = [
            "error", "exception", "fail", "timeout", "refused",
            "overflow", "out of memory", "deadlock", "crash",
            "connection reset", "broken pipe", "high memory",
            "cpu", "gc", "oom",
        ]

    def run(
        self,
        query: RCAQuery,
        prior_evidence: Optional[List[EvidenceReport]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> EvidenceReport:
        """Analyze log entries for error patterns."""
        ctx = context or {}
        data_loader: Optional[DataLoader] = ctx.get("data_loader")

        if data_loader is None:
            return EvidenceReport(
                agent_name=self.name,
                evidence_type="log",
                confidence=0.0,
            )

        from src.data_loader import _parse_date_folder
        date_folder = _parse_date_folder(query.instruction)
        if date_folder is None:
            return EvidenceReport(
                agent_name=self.name,
                evidence_type="log",
                confidence=0.0,
            )

        log_df = data_loader.load_log(query.dataset, date_folder)
        if log_df.empty:
            return EvidenceReport(
                agent_name=self.name,
                evidence_type="log",
                confidence=0.0,
            )

        analysis = self._analyze_logs(log_df, query)
        candidates = analysis.get("error_components", [])
        support = [c.get("component", "") for c in candidates]

        llm_summaries = self._generate_llm_summaries(candidates)
        for cand in candidates:
            comp = cand.get("component", "")
            if comp in llm_summaries:
                cand["llm_summary"] = llm_summaries[comp]
            else:
                cand["llm_summary"] = self._rule_based_summary(cand)

        return EvidenceReport(
            agent_name=self.name,
            evidence_type="log",
            candidates=candidates,
            confidence=analysis.get("max_error_score", 0.0),
            support=support,
            details={**analysis, "llm_summaries": llm_summaries},
        )

    def _generate_llm_summaries(
        self, error_components: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """Use LLM to generate natural language summaries for error patterns."""
        if not self.llm or not self.llm.available:
            return {}

        summaries = {}
        for comp_info in error_components[:10]:
            comp = comp_info.get("component", "")
            user_prompt = format_log_prompt(comp, comp_info)
            response = self.llm.chat_with_system(
                system_prompt=LOG_ANALYSIS_SYSTEM,
                user_prompt=user_prompt,
            )
            if response:
                summaries[comp] = response.strip()
        return summaries

    @staticmethod
    def _rule_based_summary(comp_info: Dict[str, Any]) -> str:
        """Generate a rule-based summary when LLM is unavailable."""
        comp = comp_info.get("component", "unknown")
        error_types = comp_info.get("error_types", {})
        top_errors = sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:3]
        error_str = ", ".join(f"{k}({v})" for k, v in top_errors)
        return f"{comp}: errors detected [{error_str}], ratio={comp_info.get('error_ratio', 0):.2%}"

    def _analyze_logs(self, log_df: pd.DataFrame, query: RCAQuery) -> Dict[str, Any]:
        """Analyze log entries for error patterns."""
        if log_df.empty:
            return {"error_components": [], "max_error_score": 0.0}

        log_df = log_df.copy()
        component_col = "cmdb_id" if "cmdb_id" in log_df.columns else "component"
        if component_col not in log_df.columns:
            return {"error_components": [], "max_error_score": 0.0}

        if "timestamp" in log_df.columns:
            log_df["timestamp"] = pd.to_numeric(log_df["timestamp"], errors="coerce")
            start_ts = query.start_ts
            end_ts = query.end_ts
            mask = (log_df["timestamp"] >= start_ts) & (log_df["timestamp"] <= end_ts)
            incident_logs = log_df[mask]
        else:
            incident_logs = log_df

        if incident_logs.empty:
            incident_logs = log_df

        value_col = "value" if "value" in incident_logs.columns else None
        if value_col is None:
            return {"error_components": [], "max_error_score": 0.0}

        component_errors: Dict[str, Dict[str, Any]] = {}
        for comp, group in incident_logs.groupby(component_col):
            error_count = 0
            error_types: Dict[str, int] = {}
            for val in group[value_col].dropna():
                val_str = str(val).lower()
                for kw in self.error_keywords:
                    if kw in val_str:
                        error_count += 1
                        error_types[kw] = error_types.get(kw, 0) + 1
                        break

            if error_count > 0:
                total_logs = len(group)
                error_ratio = error_count / max(total_logs, 1)
                component_errors[comp] = {
                    "component": comp,
                    "error_count": error_count,
                    "total_logs": total_logs,
                    "error_ratio": round(error_ratio, 4),
                    "error_types": error_types,
                    "anomaly_score": min(1.0, error_ratio * 3.0),
                }

        error_components = sorted(
            component_errors.values(),
            key=lambda x: x["anomaly_score"],
            reverse=True,
        )

        max_score = max((c["anomaly_score"] for c in error_components), default=0.0)

        return {
            "error_components": error_components,
            "max_error_score": round(max_score, 4),
            "total_log_entries": len(incident_logs),
        }
