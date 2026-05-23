from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.agents.base_agent import BaseAgent
from src.agents.causal_synthesis_agent import CausalSynthesisAgent
from src.agents.log_agent import LogAgent
from src.agents.reflector_agent import ReflectorAgent
from src.agents.spectral_metric_agent import SpectralMetricAgent
from src.agents.trace_agent import TraceAgent
from src.agents.verifier_agent import VerifierAgent
from src.config import SpectralRCAConfig
from src.data_loader import DataLoader
from src.llm.client import LLMClient
from src.memory.experience_store import ExperienceStore
from src.memory.pattern_evolver import PatternEvolver
from src.reasoning.abductive_engine import AbductiveReasoningEngine
from src.schemas import (
    CoordinatorState,
    EdgeEvidence,
    EvidenceReport,
    MetricAnomalyEvidence,
    RCAQuery,
    RCAResult,
    RootCauseCandidate,
    SpectralExperience,
)


class Coordinator:
    """Main coordinator for the SpectralRCA-Agent framework.

    Orchestrates the multi-agent workflow using a state machine:

        INIT -> HYPOTHESIZE -> EVIDENCE_COLLECT -> SPECTRAL_VALIDATE ->
        GRAPH_REFINE -> ABDUCTIVE_REASON -> SYNTHESIZE -> REFLECT ->
        (BACKTRACK if needed) -> FINALIZE

    Agents communicate through EvidenceReports (structured data)
    and LLM-generated natural language explanations.
    """

    def __init__(
        self,
        config: Optional[SpectralRCAConfig] = None,
        data_loader: Optional[DataLoader] = None,
    ) -> None:
        self.config = config or SpectralRCAConfig()
        self.data_loader = data_loader

        self.llm = self._create_llm_client()

        self.metric_agent = SpectralMetricAgent(self.config, llm_client=self.llm)
        self.trace_agent = TraceAgent(self.config, llm_client=self.llm)
        self.log_agent = LogAgent(self.config, llm_client=self.llm)
        self.causal_agent = CausalSynthesisAgent(self.config, llm_client=self.llm)
        self.verifier_agent = VerifierAgent(self.config)
        self.reflector_agent = ReflectorAgent(self.config, llm_client=self.llm)

        self.experience_store = ExperienceStore(self.config.memory)
        self.pattern_evolver = PatternEvolver(self.experience_store, self.config.memory)
        self.abductive_engine = AbductiveReasoningEngine(self.config, self.experience_store)

        self._state = CoordinatorState.INIT
        self._evidence_reports: List[EvidenceReport] = []
        self._anomaly_evidence: List[MetricAnomalyEvidence] = []
        self._edge_evidence: List[EdgeEvidence] = []
        self._ranked_candidates: List[RootCauseCandidate] = []
        self._trajectory: List[Dict[str, Any]] = []

    def _create_llm_client(self) -> LLMClient:
        """Create LLM client from config."""
        llm_cfg = self.config.llm
        if not llm_cfg.enable_llm or not llm_cfg.api_key:
            return LLMClient(api_key=None)

        return LLMClient(
            api_key=llm_cfg.api_key,
            base_url=llm_cfg.base_url,
            model=llm_cfg.model,
            temperature=llm_cfg.temperature,
            max_tokens=llm_cfg.max_tokens,
        )

    def run(self, query: RCAQuery) -> RCAResult:
        """Execute the full SpectralRCA-Agent pipeline for a query."""
        self._reset()
        self._log_state("INIT", "Starting SpectralRCA-Agent pipeline")

        context = self._build_context(query)

        self._state = CoordinatorState.HYPOTHESIZE
        metric_report = self.metric_agent.run(query, self._evidence_reports, context)
        self._evidence_reports.append(metric_report)
        self._anomaly_evidence = [
            MetricAnomalyEvidence(**d) for d in metric_report.details.get("anomaly_evidence", [])
        ]
        self._log_state("HYPOTHESIZE", f"Metric agent found {len(self._anomaly_evidence)} series")

        self._state = CoordinatorState.EVIDENCE_COLLECT
        trace_report = self.trace_agent.run(query, self._evidence_reports, context)
        self._evidence_reports.append(trace_report)
        log_report = self.log_agent.run(query, self._evidence_reports, context)
        self._evidence_reports.append(log_report)
        self._log_state("EVIDENCE_COLLECT", "Collected trace and log evidence")

        self._state = CoordinatorState.SPECTRAL_VALIDATE
        self._log_state("SPECTRAL_VALIDATE", "Spectral validation integrated in anomaly detection")

        self._state = CoordinatorState.GRAPH_REFINE
        causal_context = dict(context)
        causal_context["anomaly_evidence"] = self._anomaly_evidence
        causal_context["metric_series_dict"] = self._get_metric_series(query)
        causal_report = self.causal_agent.run(query, self._evidence_reports, causal_context)
        self._evidence_reports.append(causal_report)
        self._edge_evidence = [
            EdgeEvidence(**d) for d in causal_report.details.get("edge_evidence", [])
        ]
        self._ranked_candidates = [
            RootCauseCandidate(**d) for d in causal_report.details.get("root_cause_ranking", [])
        ]
        self._log_state("GRAPH_REFINE", f"Graph refined: {len(self._edge_evidence)} edges")

        self._state = CoordinatorState.ABDUCTIVE_REASON
        if self.config.enable_abductive:
            self._ranked_candidates = self.abductive_engine.reason(
                query, self._anomaly_evidence, self._edge_evidence, self._ranked_candidates,
            )
            self._trajectory.extend(self.abductive_engine.get_trajectory())
        self._log_state("ABDUCTIVE_REASON", f"Abductive reasoning complete, top: {self._ranked_candidates[0].node_id if self._ranked_candidates else 'none'}")

        self._state = CoordinatorState.SYNTHESIZE
        verify_report = self.verifier_agent.run(query, self._evidence_reports, context)
        self._evidence_reports.append(verify_report)
        self._log_state("SYNTHESIZE", "Cross-validated evidence")

        self._state = CoordinatorState.REFLECT
        reflect_report = self.reflector_agent.run(query, self._evidence_reports, context)
        self._evidence_reports.append(reflect_report)
        action = reflect_report.details.get("action", "finalize")
        self._log_state("REFLECT", f"Reflection action: {action}")

        self._state = CoordinatorState.FINALIZE
        prediction_json = self._generate_prediction(query)

        if self.config.enable_memory and self._ranked_candidates:
            self._update_experience(query)

        self._log_state("FINALIZE", "Pipeline complete")

        return RCAResult(
            prediction_json=prediction_json,
            ranked_candidates=[c.to_dict() for c in self._ranked_candidates],
            evidence_chain=[r.to_dict() for r in self._evidence_reports],
            anomaly_evidence=self._anomaly_evidence,
            edge_evidence=self._edge_evidence,
            root_cause_ranking=self._ranked_candidates,
            trajectory=self._trajectory,
            cost=self._compute_cost(),
        )

    def _reset(self) -> None:
        """Reset internal state for a new query."""
        self._state = CoordinatorState.INIT
        self._evidence_reports = []
        self._anomaly_evidence = []
        self._edge_evidence = []
        self._ranked_candidates = []
        self._trajectory = []

    def _build_context(self, query: RCAQuery) -> Dict[str, Any]:
        """Build context dict for agents."""
        return {"data_loader": self.data_loader}

    def _get_metric_series(self, query: RCAQuery) -> Dict[str, Any]:
        """Get metric series dict for causal agent."""
        if self.data_loader is None:
            return {}
        from src.data_loader import _parse_date_folder
        date_folder = _parse_date_folder(query.instruction)
        if date_folder is None:
            return {}
        return self.data_loader.build_metric_series_dict(
            query.dataset, date_folder,
            resample_interval=self.config.pipeline.resample_interval,
        )

    def _generate_prediction(self, query: RCAQuery) -> str:
        """Generate prediction JSON string from ranked candidates."""
        if not self._ranked_candidates:
            return '{"component": "unknown", "reason": "unknown", "time": "unknown"}'

        top = self._ranked_candidates[0]
        result = {
            "component": top.node_id,
            "reason": self._infer_reason(top),
            "time": top.onset_time or "unknown",
        }

        if top.explanation:
            result["explanation"] = top.explanation

        import json
        return json.dumps(result, ensure_ascii=False)

    def _infer_reason(self, candidate: RootCauseCandidate) -> str:
        """Infer root cause reason from candidate evidence."""
        for ev in self._anomaly_evidence:
            if ev.node_id == candidate.node_id and ev.anomaly_type != "no_strong_spectral_anomaly":
                reason_map = {
                    "slow_trend_anomaly": "resource_exhaustion",
                    "fast_burst_or_jitter_anomaly": "sudden_spike_or_instability",
                    "periodic_oscillation_anomaly": "periodic_interference",
                    "mixed_spectral_anomaly": "complex_failure",
                }
                return reason_map.get(ev.anomaly_type, "unknown")
        return "unknown"

    def _update_experience(self, query: RCAQuery) -> None:
        """Update experience store with current case results."""
        for ev in self._anomaly_evidence:
            if ev.final_anomaly_score > 0.3:
                experience = SpectralExperience(
                    component=ev.node_id,
                    kpi=ev.node_id.split(".")[-1] if "." in ev.node_id else "",
                    anomaly_type=ev.anomaly_type,
                    spectral_pattern={
                        "total_energy": ev.total_energy,
                        "low_ratio": ev.low_ratio,
                        "mid_ratio": ev.mid_ratio,
                        "high_ratio": ev.high_ratio,
                        "dominant_freq_energy_ratio": ev.dominant_freq_energy_ratio,
                        "spectral_entropy": ev.spectral_entropy,
                    },
                    reason=self._infer_reason_from_evidence(ev),
                    success=False,
                    timestamp=query.start_time,
                    case_id=query.task_index,
                )
                self.experience_store.add(experience)

    def _infer_reason_from_evidence(self, ev: MetricAnomalyEvidence) -> str:
        """Infer reason from anomaly evidence."""
        if ev.anomaly_type == "slow_trend_anomaly":
            return "resource_exhaustion"
        elif ev.anomaly_type == "fast_burst_or_jitter_anomaly":
            return "sudden_spike"
        elif ev.anomaly_type == "periodic_oscillation_anomaly":
            return "periodic_interference"
        return "unknown"

    def _compute_cost(self) -> Dict[str, Any]:
        """Compute cost metrics for the pipeline run."""
        return {
            "agent_calls": len(self._evidence_reports),
            "anomaly_evidence_count": len(self._anomaly_evidence),
            "edge_evidence_count": len(self._edge_evidence),
            "trajectory_steps": len(self._trajectory),
            "llm_available": self.llm.available,
        }

    def _log_state(self, state: str, description: str) -> None:
        """Log state transition."""
        self._trajectory.append({
            "state": state,
            "description": description,
            "evidence_count": len(self._evidence_reports),
        })
