from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.agents.base_agent import BaseAgent
from src.config import SpectralRCAConfig
from src.graph.graph_refiner import SpectralGraphRefinementExpert
from src.llm.client import LLMClient
from src.llm.prompts import CAUSAL_SYNTHESIS_SYSTEM, format_causal_prompt
from src.ranking.root_cause_ranker import RootCauseRanker
from src.schemas import (
    EdgeEvidence,
    EvidenceReport,
    MetricAnomalyEvidence,
    RCAQuery,
    RootCauseCandidate,
)


class CausalSynthesisAgent(BaseAgent):
    """Causal synthesis agent with LLM-enhanced reasoning explanation.

    Combines spectral graph refinement and root cause ranking,
    then uses LLM to generate natural language explanations
    for the causal reasoning chain.
    """

    def __init__(
        self,
        config: Optional[SpectralRCAConfig] = None,
        llm_client: Optional[LLMClient] = None,
    ) -> None:
        super().__init__(name="CausalSynthesisAgent", config=config)
        self.graph_expert = SpectralGraphRefinementExpert(config)
        self.ranker = RootCauseRanker()
        self.llm = llm_client

    def run(
        self,
        query: RCAQuery,
        prior_evidence: Optional[List[EvidenceReport]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> EvidenceReport:
        """Synthesize causal graph and rank root causes."""
        ctx = context or {}

        anomaly_evidence = ctx.get("anomaly_evidence", [])
        metric_series_dict = ctx.get("metric_series_dict", {})

        if not anomaly_evidence:
            metric_report = self._find_evidence(prior_evidence, "metric_spectral")
            if metric_report:
                anomaly_evidence = [
                    MetricAnomalyEvidence(**d) for d in metric_report.details.get("anomaly_evidence", [])
                ]

        edge_evidence = self.graph_expert.refine(
            anomaly_evidence,
            metric_series_dict,
            incident_start=query.start_time,
            incident_end=query.end_time,
        )

        ranked = self.ranker.rank(anomaly_evidence, edge_evidence)

        llm_explanation = self._generate_llm_explanation(
            ranked[:5], edge_evidence, query.start_time, query.end_time
        )

        for r in ranked:
            r.explanation = self._rule_based_explanation(r, anomaly_evidence)

        if llm_explanation:
            ranked[0].explanation = llm_explanation if ranked else ""

        candidates = [r.to_dict() for r in ranked[:10]]
        support = [r.node_id for r in ranked if r.root_score > 0.3]

        return EvidenceReport(
            agent_name=self.name,
            evidence_type="causal_synthesis",
            candidates=candidates,
            confidence=ranked[0].root_score if ranked else 0.0,
            support=support,
            details={
                "edge_count": len(edge_evidence),
                "kept_edges": sum(1 for e in edge_evidence if e.keep_edge),
                "ranked_count": len(ranked),
                "edge_evidence": [e.to_dict() for e in edge_evidence],
                "root_cause_ranking": [r.to_dict() for r in ranked],
                "llm_explanation": llm_explanation,
            },
        )

    def _generate_llm_explanation(
        self,
        top5: List[RootCauseCandidate],
        edge_evidence: List[EdgeEvidence],
        start_time: str,
        end_time: str,
    ) -> str:
        """Use LLM to generate natural language explanation for causal reasoning."""
        if not self.llm or not self.llm.available or not top5:
            return ""

        total_edges = len(edge_evidence)
        kept_edges = sum(1 for e in edge_evidence if e.keep_edge)
        user_prompt = format_causal_prompt(
            top5=[c.to_dict() for c in top5],
            total_edges=total_edges,
            kept_edges=kept_edges,
            start_time=start_time,
            end_time=end_time,
        )
        response = self.llm.chat_with_system(
            system_prompt=CAUSAL_SYNTHESIS_SYSTEM,
            user_prompt=user_prompt,
        )
        return response.strip() if response else ""

    @staticmethod
    def _rule_based_explanation(
        candidate: RootCauseCandidate,
        anomaly_evidence: List[MetricAnomalyEvidence],
    ) -> str:
        """Generate a rule-based explanation when LLM is unavailable."""
        anomaly_type = "unknown"
        for ev in anomaly_evidence:
            if ev.node_id == candidate.node_id:
                anomaly_type = ev.anomaly_type
                break
        return (
            f"{candidate.node_id}: root_score={candidate.root_score:.4f}, "
            f"anomaly={candidate.anomaly_score:.4f}, "
            f"out_evidence={candidate.out_evidence:.4f}, "
            f"spectral_type={anomaly_type}"
        )

    def _find_evidence(
        self,
        prior_evidence: Optional[List[EvidenceReport]],
        evidence_type: str,
    ) -> Optional[EvidenceReport]:
        """Find evidence of a specific type from prior reports."""
        if not prior_evidence:
            return None
        for report in prior_evidence:
            if report.evidence_type == evidence_type:
                return report
        return None
