from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from rca.multi_agent_rca.agents import CausalAgent, LogAgent, MetricAgent, TraceAgent, VerifierAgent
from rca.multi_agent_rca.core.schema import CoordinatorState, EvidenceReport, RCAQuery, RCAResult
from rca.multi_agent_rca.memory import MemoryStore, score_trajectory
from rca.multi_agent_rca.tools import SemanticSampler


@dataclass
class CoordinatorConfig:
    enable_sampler: bool = True
    enable_trace: bool = True
    enable_log: bool = True
    enable_verifier: bool = True
    enable_memory: bool = False
    enable_meta: bool = True
    enable_recheck_debate: bool = True
    sampler_max_traces: int = 160
    sampler_max_log_templates: int = 80
    frozen_memory: bool = True
    memory_dir: Optional[str] = None


class Coordinator:
    """MACAA-style state machine for evidence-graph RCA."""

    def __init__(self, config: Optional[CoordinatorConfig] = None) -> None:
        self.config = config or CoordinatorConfig()
        self.metric_agent = MetricAgent()
        self.trace_agent = TraceAgent()
        self.log_agent = LogAgent()
        self.causal_agent = CausalAgent(use_meta=self.config.enable_meta)
        self.verifier_agent = VerifierAgent()
        self.sampler = SemanticSampler(
            max_traces=self.config.sampler_max_traces,
            max_log_templates=self.config.sampler_max_log_templates,
        )
        self.memory = MemoryStore(
            memory_dir=None if self.config.memory_dir is None else __import__("pathlib").Path(self.config.memory_dir),
            frozen=self.config.frozen_memory,
        )

    def run(self, query: RCAQuery) -> RCAResult:
        t0 = time.time()
        trajectory: List[Dict[str, Any]] = []
        evidence_chain: List[Dict[str, Any]] = []
        self._step(trajectory, CoordinatorState.INIT, {"dataset": query.dataset, "row_id": query.row_id})

        self._step(
            trajectory,
            CoordinatorState.PRECHECK,
            {
                "time_range": [query.start_time, query.end_time],
                "target_fields": list(query.target_fields),
                "failure_count": query.failure_count,
            },
        )

        self._step(trajectory, CoordinatorState.DISPATCH, {"agent": "MetricAgent"})
        metric_report = self.metric_agent.run(query)
        evidence_chain.append(metric_report.to_dict())

        sampler_report = None
        if self.config.enable_sampler:
            sampler_report = self.sampler.run(query, metric_report)
            self._step(trajectory, CoordinatorState.DISPATCH, {"agent": "SemanticSampler", "report": sampler_report.to_dict()})

        trace_report = None
        if self.config.enable_trace:
            self._step(trajectory, CoordinatorState.DISPATCH, {"agent": "TraceAgent"})
            trace_report = self.trace_agent.run(
                query,
                metric_report,
                selected_trace_ids=sampler_report.selected_trace_ids if sampler_report else None,
            )
            evidence_chain.append(trace_report.to_dict())

        log_report = None
        if self.config.enable_log:
            self._step(trajectory, CoordinatorState.DISPATCH, {"agent": "LogAgent"})
            log_report = self.log_agent.run(
                query,
                selected_templates=sampler_report.selected_log_templates if sampler_report else None,
            )
            evidence_chain.append(log_report.to_dict())

        memory_cases = []
        if self.config.enable_memory:
            memory_cases = self.memory.retrieve(query, metric_report.details)
            self._step(trajectory, CoordinatorState.DISPATCH, {"agent": "MemoryAgent", "retrieved": len(memory_cases)})

        if self.config.enable_recheck_debate:
            self._step(trajectory, CoordinatorState.RECHECK, self._recheck(metric_report, trace_report, log_report))
            self._step(trajectory, CoordinatorState.DEBATE, self._debate(metric_report, trace_report, log_report))

        self._step(trajectory, CoordinatorState.SYNTHESIZE, {"agent": "CausalAgent"})
        graph, ranked = self.causal_agent.run(query, metric_report, trace_report, log_report, memory_cases)
        prediction = self.causal_agent.to_prediction_json(query, ranked)

        self._step(trajectory, CoordinatorState.REFLECT, self._reflect(query, ranked, metric_report, trace_report, log_report))
        if self.config.enable_verifier:
            prediction = self.verifier_agent.run(query, prediction, ranked)
            self._step(trajectory, CoordinatorState.FINALIZE, {"agent": "VerifierAgent", "verified": True})
        else:
            self._step(trajectory, CoordinatorState.FINALIZE, {"agent": "VerifierAgent", "verified": False})

        cost = {
            "elapsed_seconds": round(time.time() - t0, 4),
            "llm_calls": 0,
            "trajectory_quality": score_trajectory(trajectory),
        }
        diagnostics = {
            "sampler": sampler_report.to_dict() if sampler_report else None,
            "config": self.config.__dict__,
        }
        return RCAResult(
            prediction_json=prediction,
            ranked_candidates=ranked,
            evidence_chain=evidence_chain,
            graph=graph,
            trajectory=trajectory,
            cost=cost,
            diagnostics=diagnostics,
        )

    def record_memory(self, query: RCAQuery, result: RCAResult, score: Optional[float]) -> None:
        self.memory.record(query, result, score)

    def _step(self, trajectory: List[Dict[str, Any]], state: CoordinatorState, payload: Dict[str, Any]) -> None:
        trajectory.append({"state": state.value, "payload": payload})

    def _recheck(
        self,
        metric_report: EvidenceReport,
        trace_report: Optional[EvidenceReport],
        log_report: Optional[EvidenceReport],
    ) -> Dict[str, Any]:
        notes = []
        if metric_report.confidence < 0.15:
            notes.append("Metric confidence is low; final answer should rely more on trace/log if available.")
        if trace_report and not trace_report.candidates:
            notes.append("Trace report has no ranked propagation candidate.")
        if log_report and log_report.confidence < 0.1:
            notes.append("Log evidence is weak or unavailable.")
        return {"notes": notes or ["No extra recheck needed."]}

    def _debate(
        self,
        metric_report: EvidenceReport,
        trace_report: Optional[EvidenceReport],
        log_report: Optional[EvidenceReport],
    ) -> Dict[str, Any]:
        metric_top = set(list(metric_report.details.get("component_scores", {}).keys())[:5])
        trace_top = set()
        if trace_report:
            trace_top = set(list(trace_report.details.get("component_scores", {}).keys())[:5])
        conflict = bool(metric_top and trace_top and not (metric_top & trace_top))
        return {
            "conflict": conflict,
            "resolution": "Trace can only promote components that also have metric/log support."
            if conflict
            else "Metric, trace, and log evidence are compatible enough for synthesis.",
        }

    def _reflect(
        self,
        query: RCAQuery,
        ranked: List[Dict[str, Any]],
        metric_report: EvidenceReport,
        trace_report: Optional[EvidenceReport],
        log_report: Optional[EvidenceReport],
    ) -> Dict[str, Any]:
        missing = []
        if "root cause reason" in query.target_fields and not ranked[0].get("reason"):
            missing.append("reason")
        if "root cause component" in query.target_fields and not ranked[0].get("component"):
            missing.append("component")
        if "root cause occurrence datetime" in query.target_fields and not ranked[0].get("occurrence_time"):
            missing.append("time")
        return {
            "missing_fields": missing,
            "top_candidate": ranked[0] if ranked else {},
            "evidence_confidence": {
                "metric": metric_report.confidence,
                "trace": trace_report.confidence if trace_report else 0.0,
                "log": log_report.confidence if log_report else 0.0,
            },
        }
