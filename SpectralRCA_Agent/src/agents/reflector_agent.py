from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.agents.base_agent import BaseAgent
from src.config import SpectralRCAConfig
from src.llm.client import LLMClient
from src.llm.prompts import REFLECTOR_SYSTEM, format_reflector_prompt
from src.schemas import EvidenceReport, RCAQuery


class ReflectorAgent(BaseAgent):
    """Reflection agent with LLM-enhanced diagnosis quality evaluation.

    Evaluates diagnosis quality and triggers backtracking when needed.
    When LLM is available, it generates natural language assessment
    of the diagnosis reliability and improvement suggestions.
    """

    def __init__(
        self,
        config: Optional[SpectralRCAConfig] = None,
        llm_client: Optional[LLMClient] = None,
        confidence_threshold: float = 0.5,
    ) -> None:
        super().__init__(name="ReflectorAgent", config=config)
        self.llm = llm_client
        self.confidence_threshold = confidence_threshold

    def run(
        self,
        query: RCAQuery,
        prior_evidence: Optional[List[EvidenceReport]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> EvidenceReport:
        """Evaluate diagnosis quality and suggest improvements."""
        if not prior_evidence:
            return EvidenceReport(
                agent_name=self.name,
                evidence_type="reflection",
                confidence=0.0,
                details={"action": "need_more_evidence"},
            )

        max_confidence = max((r.confidence for r in prior_evidence), default=0.0)
        total_support = set()
        for report in prior_evidence:
            total_support.update(report.support)

        causal_report = None
        for report in prior_evidence:
            if report.evidence_type == "causal_synthesis":
                causal_report = report
                break

        action, suggestions = self._evaluate_rule_based(
            max_confidence, total_support, causal_report
        )

        llm_assessment = self._generate_llm_assessment(
            max_confidence, total_support, prior_evidence, causal_report
        )

        if llm_assessment:
            suggestions.append(f"LLM Assessment: {llm_assessment}")

        return EvidenceReport(
            agent_name=self.name,
            evidence_type="reflection",
            confidence=max_confidence,
            candidates=[],
            details={
                "action": action,
                "max_confidence": round(max_confidence, 4),
                "total_supported_nodes": len(total_support),
                "suggestions": suggestions,
                "llm_assessment": llm_assessment,
            },
        )

    def _evaluate_rule_based(
        self,
        max_confidence: float,
        total_support: set,
        causal_report: Optional[EvidenceReport],
    ) -> tuple:
        """Rule-based evaluation for diagnosis quality."""
        action = "finalize"
        suggestions: List[str] = []

        if max_confidence < self.confidence_threshold:
            action = "backtrack"
            suggestions.append("Low confidence - consider expanding search scope")
        elif len(total_support) == 0:
            action = "backtrack"
            suggestions.append("No supported candidates - re-examine evidence")
        elif causal_report and len(causal_report.candidates) > 0:
            top_score = causal_report.candidates[0].get("root_score", 0.0)
            if top_score < 0.3:
                action = "re_hypothesize"
                suggestions.append("Top candidate has low score - generate new hypotheses")

        if action == "finalize" and causal_report:
            suggestions.append("Diagnosis is confident - proceed to finalization")

        return action, suggestions

    def _generate_llm_assessment(
        self,
        max_confidence: float,
        total_support: set,
        prior_evidence: List[EvidenceReport],
        causal_report: Optional[EvidenceReport],
    ) -> str:
        """Use LLM to evaluate diagnosis quality and suggest improvements."""
        if not self.llm or not self.llm.available:
            return ""

        evidence_types = [r.evidence_type for r in prior_evidence]
        top1_text = ""
        top1_score = 0.0
        if causal_report and causal_report.candidates:
            top1 = causal_report.candidates[0]
            top1_text = f"{top1.get('node_id', '?')}: score={top1.get('root_score', 0):.4f}"
            top1_score = top1.get("root_score", 0.0)

        user_prompt = format_reflector_prompt(
            max_confidence=max_confidence,
            supported_nodes=len(total_support),
            evidence_types=evidence_types,
            top1_text=top1_text,
            top1_score=top1_score,
        )
        response = self.llm.chat_with_system(
            system_prompt=REFLECTOR_SYSTEM,
            user_prompt=user_prompt,
        )
        return response.strip() if response else ""
