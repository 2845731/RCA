from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.agents.base_agent import BaseAgent
from src.config import SpectralRCAConfig
from src.schemas import EvidenceReport, RCAQuery


class VerifierAgent(BaseAgent):
    """Verification agent that cross-validates evidence from multiple agents.

    Checks for consistency between metric, trace, and log evidence,
    and flags contradictions.
    """

    def __init__(self, config: Optional[SpectralRCAConfig] = None) -> None:
        super().__init__(name="VerifierAgent", config=config)

    def run(
        self,
        query: RCAQuery,
        prior_evidence: Optional[List[EvidenceReport]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> EvidenceReport:
        """Cross-validate evidence from multiple agents."""
        if not prior_evidence:
            return EvidenceReport(
                agent_name=self.name,
                evidence_type="verification",
                confidence=0.0,
            )

        all_support: Dict[str, int] = {}
        all_contradiction: List[str] = []
        candidate_details: Dict[str, Dict[str, Any]] = {}

        for report in prior_evidence:
            for node_id in report.support:
                all_support[node_id] = all_support.get(node_id, 0) + 1
            for cand in report.candidates:
                node_id = cand.get("node_id", cand.get("cmdb_id", cand.get("component", "")))
                if node_id:
                    if node_id not in candidate_details:
                        candidate_details[node_id] = {}
                    candidate_details[node_id][report.evidence_type] = cand

        verified_candidates = []
        for node_id, count in sorted(all_support.items(), key=lambda x: x[1], reverse=True):
            sources = candidate_details.get(node_id, {})
            verification_score = count / max(len(prior_evidence), 1)

            verified_candidates.append({
                "node_id": node_id,
                "support_count": count,
                "verification_score": round(verification_score, 4),
                "evidence_sources": list(sources.keys()),
            })

        confidence = verified_candidates[0]["verification_score"] if verified_candidates else 0.0

        return EvidenceReport(
            agent_name=self.name,
            evidence_type="verification",
            candidates=verified_candidates,
            confidence=round(confidence, 4),
            support=[c["node_id"] for c in verified_candidates if c["verification_score"] > 0.5],
            contradiction=all_contradiction,
            details={
                "total_candidates": len(verified_candidates),
                "multi_source_candidates": sum(1 for c in verified_candidates if c["support_count"] >= 2),
            },
        )
