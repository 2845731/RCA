from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.agents.base_agent import BaseAgent
from src.anomaly.metric_anomaly_expert import MetricAnomalyExpert
from src.config import SpectralRCAConfig
from src.data_loader import DataLoader
from src.llm.client import LLMClient
from src.llm.prompts import (
    SPECTRAL_METRIC_SYSTEM,
    format_spectral_metric_prompt,
)
from src.preprocessing import extract_windows, fill_missing
from src.schemas import EvidenceReport, MetricAnomalyEvidence, RCAQuery


class SpectralMetricAgent(BaseAgent):
    """Spectral-enhanced metric anomaly detection agent with LLM explanation.

    Innovation 1: Combines traditional anomaly detection with DFT-based
    spectral analysis to identify frequency-domain anomaly patterns
    (slow_trend, fast_burst, periodic_oscillation, mixed_spectral).

    When LLM is available, it generates natural language explanations
    for each detected anomaly. When LLM is unavailable, it falls back
    to rule-based explanation.
    """

    def __init__(
        self,
        config: Optional[SpectralRCAConfig] = None,
        llm_client: Optional[LLMClient] = None,
    ) -> None:
        super().__init__(name="SpectralMetricAgent", config=config)
        self.expert = MetricAnomalyExpert(config)
        self.llm = llm_client

    def run(
        self,
        query: RCAQuery,
        prior_evidence: Optional[List[EvidenceReport]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> EvidenceReport:
        """Run spectral-enhanced anomaly detection on all metric series."""
        ctx = context or {}
        data_loader: Optional[DataLoader] = ctx.get("data_loader")

        if data_loader is None:
            return EvidenceReport(
                agent_name=self.name,
                evidence_type="metric_spectral",
                confidence=0.0,
                details={"error": "No data_loader in context"},
            )

        from src.data_loader import _parse_date_folder
        date_folder = _parse_date_folder(query.instruction)
        if date_folder is None:
            return EvidenceReport(
                agent_name=self.name,
                evidence_type="metric_spectral",
                confidence=0.0,
                details={"error": "Cannot parse date from instruction"},
            )

        metric_series_dict = data_loader.build_metric_series_dict(
            query.dataset, date_folder,
            resample_interval=self.config.pipeline.resample_interval if self.config else "60s",
        )

        if not metric_series_dict:
            return EvidenceReport(
                agent_name=self.name,
                evidence_type="metric_spectral",
                confidence=0.0,
                details={"error": "No metric series loaded"},
            )

        anomaly_evidence = self.expert.detect(
            metric_series_dict,
            incident_start=query.start_time,
            incident_end=query.end_time,
            baseline_start=self._compute_baseline_start(query),
            baseline_end=query.start_time,
        )

        high_conf_evidence = [ev for ev in anomaly_evidence if ev.final_anomaly_score > 0.5]
        llm_explanations = self._generate_llm_explanations(high_conf_evidence[:10])

        for ev in anomaly_evidence:
            if ev.node_id in llm_explanations:
                ev.explanation = llm_explanations[ev.node_id]
            elif ev.final_anomaly_score > 0.5:
                ev.explanation = self._rule_based_explanation(ev)

        candidates = []
        support = []
        for ev in anomaly_evidence:
            if ev.final_anomaly_score > 0.3:
                candidates.append(ev.to_dict())
                support.append(ev.node_id)

        confidence = max((ev.final_anomaly_score for ev in anomaly_evidence), default=0.0)

        return EvidenceReport(
            agent_name=self.name,
            evidence_type="metric_spectral",
            candidates=candidates,
            confidence=round(confidence, 4),
            support=support,
            details={
                "total_series": len(metric_series_dict),
                "anomalous_count": len(candidates),
                "anomaly_evidence": [ev.to_dict() for ev in anomaly_evidence],
                "llm_explanations": llm_explanations,
            },
        )

    def _generate_llm_explanations(
        self, evidence_list: List[MetricAnomalyEvidence]
    ) -> Dict[str, str]:
        """Use LLM to generate natural language explanations for anomalies."""
        if not self.llm or not self.llm.available:
            return {}

        explanations = {}
        for ev in evidence_list:
            user_prompt = format_spectral_metric_prompt(ev.to_dict())
            response = self.llm.chat_with_system(
                system_prompt=SPECTRAL_METRIC_SYSTEM,
                user_prompt=user_prompt,
            )
            if response:
                explanations[ev.node_id] = response.strip()
        return explanations

    @staticmethod
    def _rule_based_explanation(ev: MetricAnomalyEvidence) -> str:
        """Generate a rule-based explanation when LLM is unavailable."""
        type_desc = {
            "slow_trend_anomaly": "low-freq energy dominant, likely resource exhaustion (memory leak/disk full)",
            "fast_burst_or_jitter_anomaly": "high-freq energy dominant, likely sudden spike (crash/network jitter)",
            "periodic_oscillation_anomaly": "dominant frequency detected, likely periodic interference (cron job/config error)",
            "mixed_spectral_anomaly": "multi-band anomaly, complex failure pattern",
            "no_strong_spectral_anomaly": "no strong spectral pattern, mild fluctuation",
        }
        desc = type_desc.get(ev.anomaly_type, "unknown pattern")
        return f"{ev.node_id}: {desc}, score={ev.final_anomaly_score:.4f}"

    def run_standalone(
        self,
        dataset_dir: str,
        dataset: str,
        date_folder: str,
        incident_start: str,
        incident_end: str,
    ) -> Dict[str, Any]:
        """Run anomaly detection standalone for ablation evaluation."""
        data_loader = DataLoader(dataset_dir)
        metric_series_dict = data_loader.build_metric_series_dict(
            dataset, date_folder,
            resample_interval=self.config.pipeline.resample_interval if self.config else "60s",
        )

        baseline_start = self._compute_baseline_start_from_times(incident_start, incident_end)

        anomaly_evidence = self.expert.detect(
            metric_series_dict,
            incident_start=incident_start,
            incident_end=incident_end,
            baseline_start=baseline_start,
            baseline_end=incident_start,
        )

        total = len(anomaly_evidence)
        high_conf = sum(1 for e in anomaly_evidence if e.final_anomaly_score >= 0.85)
        candidate = sum(1 for e in anomaly_evidence if e.final_anomaly_score >= 0.70)
        spectral_types = {}
        for e in anomaly_evidence:
            if e.final_anomaly_score > 0.3:
                spectral_types[e.anomaly_type] = spectral_types.get(e.anomaly_type, 0) + 1

        return {
            "anomaly_evidence": [ev.to_dict() for ev in anomaly_evidence],
            "summary": {
                "total_series": total,
                "high_confidence_anomalies": high_conf,
                "candidate_anomalies": candidate,
                "spectral_type_distribution": spectral_types,
            },
        }

    def _compute_baseline_start(self, query: RCAQuery) -> str:
        """Compute baseline start time from query."""
        from datetime import datetime, timedelta
        inc_start = datetime.strptime(query.start_time, "%Y-%m-%d %H:%M:%S")
        inc_end = datetime.strptime(query.end_time, "%Y-%m-%d %H:%M:%S")
        duration = inc_end - inc_start
        multiplier = self.config.pipeline.baseline_window_multiplier if self.config else 3
        baseline_start = inc_start - multiplier * duration
        return baseline_start.strftime("%Y-%m-%d %H:%M:%S")

    def _compute_baseline_start_from_times(self, incident_start: str, incident_end: str) -> str:
        """Compute baseline start time from time strings."""
        from datetime import datetime, timedelta
        inc_start = datetime.strptime(incident_start, "%Y-%m-%d %H:%M:%S")
        inc_end = datetime.strptime(incident_end, "%Y-%m-%d %H:%M:%S")
        duration = inc_end - inc_start
        multiplier = self.config.pipeline.baseline_window_multiplier if self.config else 3
        baseline_start = inc_start - multiplier * duration
        return baseline_start.strftime("%Y-%m-%d %H:%M:%S")
