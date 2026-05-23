from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.schemas import AbductiveHypothesis, BeliefState, MetricAnomalyEvidence


class BeliefStateManager:
    """Manage structured belief states for abductive reasoning.

    Implements the belief state graph from Innovation 1:
    Spectral-Enhanced Abductive Reasoning.

    Belief state transitions:
        EMPTY → HYPOTHESIZED → EVIDENCE_SUPPORTED → SPECTRAL_CONFIRMED → VALIDATED
                               → EVIDENCE_CONTRADICTED → SPECTRAL_REJECTED → REJECTED
                               → BACKTRACK → HYPOTHESIZED (new hypothesis)
    """

    def __init__(self) -> None:
        self._hypotheses: Dict[str, AbductiveHypothesis] = {}
        self._iteration: int = 0

    def create_hypothesis(
        self,
        component: str,
        reason: Optional[str] = None,
        spectral_anomaly_type: Optional[str] = None,
    ) -> AbductiveHypothesis:
        """Create a new hypothesis in HYPOTHESIZED state."""
        hypothesis = AbductiveHypothesis(
            component=component,
            reason=reason,
            spectral_anomaly_type=spectral_anomaly_type,
            belief_state=BeliefState.HYPOTHESIZED,
            evidence_for=[],
            evidence_against=[],
            confidence=0.0,
            iteration=self._iteration,
        )
        self._hypotheses[component] = hypothesis
        return hypothesis

    def update_with_evidence(
        self,
        component: str,
        evidence_for: Optional[List[str]] = None,
        evidence_against: Optional[List[str]] = None,
    ) -> Optional[AbductiveHypothesis]:
        """Update hypothesis belief state based on evidence."""
        hypothesis = self._hypotheses.get(component)
        if hypothesis is None:
            return None

        if evidence_for:
            hypothesis.evidence_for.extend(evidence_for)
        if evidence_against:
            hypothesis.evidence_against.extend(evidence_against)

        if len(hypothesis.evidence_against) > len(hypothesis.evidence_for):
            hypothesis.belief_state = BeliefState.EVIDENCE_CONTRADICTED
        else:
            hypothesis.belief_state = BeliefState.EVIDENCE_SUPPORTED

        self._update_confidence(hypothesis)
        return hypothesis

    def confirm_with_spectral(
        self,
        component: str,
        spectral_anomaly_type: str,
    ) -> Optional[AbductiveHypothesis]:
        """Confirm hypothesis with spectral evidence.

        This is the key step of Innovation 1: spectral anomaly types
        serve as structured belief states that constrain reasoning direction.
        """
        hypothesis = self._hypotheses.get(component)
        if hypothesis is None:
            return None

        hypothesis.spectral_anomaly_type = spectral_anomaly_type

        if spectral_anomaly_type != "no_strong_spectral_anomaly":
            hypothesis.belief_state = BeliefState.SPECTRAL_CONFIRMED
            hypothesis.evidence_for.append(f"spectral_confirmed:{spectral_anomaly_type}")
        else:
            hypothesis.belief_state = BeliefState.SPECTRAL_REJECTED
            hypothesis.evidence_against.append("spectral_rejected:no_anomaly")

        self._update_confidence(hypothesis)
        return hypothesis

    def validate_hypothesis(self, component: str) -> Optional[AbductiveHypothesis]:
        """Mark a hypothesis as validated (final state)."""
        hypothesis = self._hypotheses.get(component)
        if hypothesis is None:
            return None

        if hypothesis.belief_state == BeliefState.SPECTRAL_CONFIRMED:
            hypothesis.belief_state = BeliefState.VALIDATED
        elif hypothesis.belief_state == BeliefState.SPECTRAL_REJECTED:
            hypothesis.belief_state = BeliefState.REJECTED
        elif hypothesis.belief_state == BeliefState.EVIDENCE_SUPPORTED:
            hypothesis.belief_state = BeliefState.VALIDATED

        self._update_confidence(hypothesis)
        return hypothesis

    def backtrack(self, component: str) -> Optional[AbductiveHypothesis]:
        """Backtrack from a hypothesis, marking it for re-evaluation."""
        hypothesis = self._hypotheses.get(component)
        if hypothesis is None:
            return None

        hypothesis.belief_state = BeliefState.HYPOTHESIZED
        hypothesis.evidence_for = []
        hypothesis.evidence_against = []
        hypothesis.confidence = 0.0
        hypothesis.iteration = self._iteration + 1
        return hypothesis

    def get_hypotheses_by_state(self, state: BeliefState) -> List[AbductiveHypothesis]:
        """Get all hypotheses in a specific belief state."""
        return [h for h in self._hypotheses.values() if h.belief_state == state]

    def get_top_hypotheses(self, k: int = 5) -> List[AbductiveHypothesis]:
        """Get top-k hypotheses sorted by confidence."""
        sorted_h = sorted(
            self._hypotheses.values(),
            key=lambda h: h.confidence,
            reverse=True,
        )
        return sorted_h[:k]

    def advance_iteration(self) -> int:
        """Advance to the next reasoning iteration."""
        self._iteration += 1
        return self._iteration

    def get_state_summary(self) -> Dict[str, Any]:
        """Get a summary of all belief states."""
        state_counts: Dict[str, int] = {}
        for h in self._hypotheses.values():
            key = h.belief_state.value
            state_counts[key] = state_counts.get(key, 0) + 1

        return {
            "iteration": self._iteration,
            "total_hypotheses": len(self._hypotheses),
            "state_distribution": state_counts,
        }

    def _update_confidence(self, hypothesis: AbductiveHypothesis) -> None:
        """Update hypothesis confidence based on evidence and belief state."""
        base = 0.0

        if hypothesis.belief_state == BeliefState.VALIDATED:
            base = 0.9
        elif hypothesis.belief_state == BeliefState.SPECTRAL_CONFIRMED:
            base = 0.7
        elif hypothesis.belief_state == BeliefState.EVIDENCE_SUPPORTED:
            base = 0.5
        elif hypothesis.belief_state == BeliefState.HYPOTHESIZED:
            base = 0.2
        elif hypothesis.belief_state == BeliefState.EVIDENCE_CONTRADICTED:
            base = 0.1
        elif hypothesis.belief_state == BeliefState.SPECTRAL_REJECTED:
            base = 0.05
        elif hypothesis.belief_state == BeliefState.REJECTED:
            base = 0.0

        evidence_bonus = min(0.1, len(hypothesis.evidence_for) * 0.02)
        evidence_penalty = min(0.3, len(hypothesis.evidence_against) * 0.05)

        hypothesis.confidence = max(0.0, min(1.0, base + evidence_bonus - evidence_penalty))
