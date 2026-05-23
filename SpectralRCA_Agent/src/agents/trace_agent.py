from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.agents.base_agent import BaseAgent
from src.config import SpectralRCAConfig
from src.data_loader import DataLoader
from src.llm.client import LLMClient
from src.llm.prompts import TRACE_ANALYSIS_SYSTEM, format_trace_prompt
from src.schemas import EvidenceReport, RCAQuery


class TraceAgent(BaseAgent):
    """Trace analysis agent with LLM-enhanced call chain interpretation.

    Innovation 3: Decomposes diagnosis along trace graph topology,
    assigning spectral-enhanced analysis to each span.
    When LLM is available, it generates natural language analysis
    of the call chain latency patterns.
    """

    def __init__(
        self,
        config: Optional[SpectralRCAConfig] = None,
        llm_client: Optional[LLMClient] = None,
    ) -> None:
        super().__init__(name="TraceAgent", config=config)
        self.llm = llm_client

    def run(
        self,
        query: RCAQuery,
        prior_evidence: Optional[List[EvidenceReport]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> EvidenceReport:
        """Analyze trace spans for anomaly patterns."""
        ctx = context or {}
        data_loader: Optional[DataLoader] = ctx.get("data_loader")

        if data_loader is None:
            return EvidenceReport(
                agent_name=self.name,
                evidence_type="trace",
                confidence=0.0,
            )

        from src.data_loader import _parse_date_folder
        date_folder = _parse_date_folder(query.instruction)
        if date_folder is None:
            return EvidenceReport(
                agent_name=self.name,
                evidence_type="trace",
                confidence=0.0,
            )

        trace_df = data_loader.load_trace(query.dataset, date_folder)
        if trace_df.empty:
            return EvidenceReport(
                agent_name=self.name,
                evidence_type="trace",
                confidence=0.0,
            )

        span_analysis = self._analyze_spans(trace_df, query)
        candidates = span_analysis.get("anomalous_spans", [])
        support = [s.get("cmdb_id", "") for s in candidates if s.get("cmdb_id")]

        llm_analysis = self._generate_llm_analysis(
            span_analysis.get("anomalous_spans", []),
            span_analysis.get("call_chains", []),
        )

        return EvidenceReport(
            agent_name=self.name,
            evidence_type="trace",
            candidates=candidates,
            confidence=span_analysis.get("max_anomaly", 0.0),
            support=support,
            details={**span_analysis, "llm_analysis": llm_analysis},
        )

    def _generate_llm_analysis(
        self,
        anomalous_spans: List[Dict[str, Any]],
        call_chains: List[Dict[str, Any]],
    ) -> str:
        """Use LLM to generate natural language analysis of trace patterns."""
        if not self.llm or not self.llm.available:
            return ""

        user_prompt = format_trace_prompt(anomalous_spans, call_chains)
        response = self.llm.chat_with_system(
            system_prompt=TRACE_ANALYSIS_SYSTEM,
            user_prompt=user_prompt,
        )
        return response.strip() if response else ""

    def _analyze_spans(self, trace_df: pd.DataFrame, query: RCAQuery) -> Dict[str, Any]:
        """Analyze trace spans for latency anomalies and call chain patterns."""
        if trace_df.empty:
            return {"anomalous_spans": [], "max_anomaly": 0.0}

        trace_df = trace_df.copy()
        if "duration" in trace_df.columns:
            trace_df["duration"] = pd.to_numeric(trace_df["duration"], errors="coerce")
        if "timestamp" in trace_df.columns:
            trace_df["timestamp"] = pd.to_numeric(trace_df["timestamp"], errors="coerce")

        start_ts_ms = query.start_ts * 1000
        end_ts_ms = query.end_ts * 1000

        if "timestamp" in trace_df.columns:
            mask = (trace_df["timestamp"] >= start_ts_ms) & (trace_df["timestamp"] <= end_ts_ms)
            incident_traces = trace_df[mask]
        else:
            incident_traces = trace_df

        if incident_traces.empty:
            incident_traces = trace_df

        span_stats = self._compute_span_statistics(incident_traces)
        anomalous_spans = self._identify_anomalous_spans(span_stats)

        call_chains = self._extract_call_chains(incident_traces)

        max_anomaly = max((s.get("anomaly_score", 0.0) for s in anomalous_spans), default=0.0)

        return {
            "anomalous_spans": anomalous_spans,
            "call_chains": call_chains,
            "span_statistics": span_stats,
            "max_anomaly": round(max_anomaly, 4),
        }

    def _compute_span_statistics(self, trace_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Compute per-component span statistics."""
        if trace_df.empty or "cmdb_id" not in trace_df.columns:
            return []

        stats = []
        for cmdb_id, group in trace_df.groupby("cmdb_id"):
            durations = group["duration"].dropna().values if "duration" in group.columns else np.array([])
            if len(durations) == 0:
                continue

            stats.append({
                "cmdb_id": cmdb_id,
                "span_count": len(group),
                "mean_duration": float(np.mean(durations)),
                "median_duration": float(np.median(durations)),
                "p95_duration": float(np.percentile(durations, 95)) if len(durations) >= 5 else float(np.max(durations)),
                "max_duration": float(np.max(durations)),
                "std_duration": float(np.std(durations)),
            })

        return stats

    def _identify_anomalous_spans(self, span_stats: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify spans with anomalous latency patterns."""
        if not span_stats:
            return []

        all_medians = [s["median_duration"] for s in span_stats]
        if not all_medians:
            return []

        global_median = np.median(all_medians)
        global_std = np.std(all_medians)
        if global_std < 1e-6:
            return []

        anomalous = []
        for stat in span_stats:
            z_score = (stat["median_duration"] - global_median) / global_std
            if z_score > 2.0:
                anomalous.append({
                    "cmdb_id": stat["cmdb_id"],
                    "anomaly_score": min(1.0, z_score / 5.0),
                    "median_duration": stat["median_duration"],
                    "z_score": round(z_score, 4),
                    "reason": "elevated_latency",
                })

        return anomalous

    def _extract_call_chains(self, trace_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Extract call chain structure from trace spans."""
        if trace_df.empty or "trace_id" not in trace_df.columns:
            return []

        chains = []
        for trace_id, group in trace_df.groupby("trace_id"):
            if "parent_id" not in group.columns or "span_id" not in group.columns:
                continue

            components = list(group["cmdb_id"].unique()) if "cmdb_id" in group.columns else []
            if len(components) >= 2:
                chains.append({
                    "trace_id": trace_id,
                    "components": components,
                    "depth": len(components),
                })

            if len(chains) >= 50:
                break

        return chains
