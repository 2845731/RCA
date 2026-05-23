from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.config import ReasoningConfig, SpectralRCAConfig
from src.memory.experience_store import ExperienceStore
from src.memory.pattern_evolver import PatternEvolver
from src.reasoning.belief_state import BeliefStateManager
from src.schemas import (
    AbductiveHypothesis,
    BeliefState,
    EdgeEvidence,
    MetricAnomalyEvidence,
    RCAQuery,
    RootCauseCandidate,
)


class AbductiveReasoningEngine:
    """Spectral-enhanced abductive reasoning engine.

    Implements Innovation 1: Frequency-Domain Enhanced Abductive Reasoning.

    Reasoning loop:
        1. HYPOTHESIZE: Generate candidate root cause hypotheses
           based on anomaly evidence and spectral anomaly types.
        2. EVIDENCE_COLLECT: Gather supporting/contradicting evidence
           from metric, trace, and log agents.
        3. SPECTRAL_VALIDATE: Validate hypotheses using spectral
           anomaly types as structured belief states.
        4. GRAPH_REFINE: Refine causal graph with spectral edge validation.
        5. ABDUCTIVE_REASON: Select best hypothesis; if confidence is low,
           BACKTRACK and generate new hypotheses.
        6. SYNTHESIZE: Produce final root cause ranking.

    This is inspired by Graph of States (GoS) abductive reasoning,
    but with spectral anomaly types as the structured belief states.
    """

    def __init__(
        self,
        config: Optional[SpectralRCAConfig] = None,
        experience_store: Optional[ExperienceStore] = None,
    ) -> None:
        self.config = config or SpectralRCAConfig()
        self.reasoning_cfg = self.config.reasoning
        self.belief_manager = BeliefStateManager()
        self.experience_store = experience_store
        self.pattern_evolver = (
            PatternEvolver(experience_store, self.config.memory)
            if experience_store
            else None
        )
        self._trajectory: List[Dict[str, Any]] = []

    def reason(
        self,
        query: RCAQuery,
        anomaly_evidence: List[MetricAnomalyEvidence],
        edge_evidence: List[EdgeEvidence],
        ranked_candidates: List[RootCauseCandidate],
    ) -> List[RootCauseCandidate]:
        """Run the abductive reasoning loop.

        Args:
            query: The RCA query.
            anomaly_evidence: Spectral-enhanced anomaly evidence.
            edge_evidence: Spectral-validated edge evidence.
            ranked_candidates: Initial root cause ranking.

        Returns:
            Refined root cause ranking after abductive reasoning.
        """
        self._trajectory = []
        self.belief_manager = BeliefStateManager()

        self._log_step("INIT", "Starting abductive reasoning", {
            "anomaly_count": len(anomaly_evidence),
            "edge_count": len(edge_evidence),
            "candidate_count": len(ranked_candidates),
        })

        self._hypothesize(anomaly_evidence, ranked_candidates)

        for iteration in range(self.reasoning_cfg.max_abductive_iterations):
            self.belief_manager.advance_iteration()

            self._collect_evidence(anomaly_evidence, edge_evidence)

            if self.config.enable_spectral_belief:
                self._spectral_validate(anomaly_evidence)

            self._abductive_evaluate()

            top = self.belief_manager.get_top_hypotheses(1)
            if top and top[0].confidence >= self.reasoning_cfg.backtrack_threshold:
                break

            if self.reasoning_cfg.enable_backtrack:
                self._backtrack_low_confidence()

        self._log_step("SYNTHESIZE", "Synthesizing final results", {})

        refined_ranking = self._synthesize(ranked_candidates)

        self._log_step("FINALIZE", "Abductive reasoning complete", {
            "final_top_component": refined_ranking[0].node_id if refined_ranking else None,
            "final_top_score": refined_ranking[0].root_score if refined_ranking else 0.0,
        })

        return refined_ranking

    def get_trajectory(self) -> List[Dict[str, Any]]:
        """Get the reasoning trajectory for analysis."""
        return self._trajectory

    def _hypothesize(
        self,
        anomaly_evidence: List[MetricAnomalyEvidence],
        ranked_candidates: List[RootCauseCandidate],
    ) -> None:
        """Generate initial hypotheses from anomaly evidence."""
        sorted_evidence = sorted(anomaly_evidence, key=lambda e: e.final_anomaly_score, reverse=True)

        top_k = self.reasoning_cfg.hypothesis_top_k
        for ev in sorted_evidence[:top_k]:
            if ev.final_anomaly_score < 0.1:
                continue

            reason = self._infer_reason_from_anomaly_type(ev.anomaly_type)
            self.belief_manager.create_hypothesis(
                component=ev.node_id,
                reason=reason,
                spectral_anomaly_type=ev.anomaly_type,
            )

        for cand in ranked_candidates[:3]:
            if cand.node_id not in self.belief_manager._hypotheses:
                self.belief_manager.create_hypothesis(
                    component=cand.node_id,
                    reason=None,
                )

        self._log_step("HYPOTHESIZE", "Generated hypotheses", {
            "hypothesis_count": len(self.belief_manager._hypotheses),
        })

    def _collect_evidence(
        self,
        anomaly_evidence: List[MetricAnomalyEvidence],
        edge_evidence: List[EdgeEvidence],
    ) -> None:
        """Collect evidence for/against each hypothesis."""
        node_anomaly = {e.node_id: e for e in anomaly_evidence}
        outgoing_edges: Dict[str, List[EdgeEvidence]] = {}
        incoming_edges: Dict[str, List[EdgeEvidence]] = {}

        for edge in edge_evidence:
            if edge.keep_edge:
                outgoing_edges.setdefault(edge.source, []).append(edge)
                incoming_edges.setdefault(edge.target, []).append(edge)

        for comp, hypothesis in self.belief_manager._hypotheses.items():
            evidence_for = []
            evidence_against = []

            anomaly_ev = node_anomaly.get(comp)
            if anomaly_ev:
                if anomaly_ev.final_anomaly_score > 0.5:
                    evidence_for.append(f"high_anomaly_score:{anomaly_ev.final_anomaly_score:.2f}")
                elif anomaly_ev.final_anomaly_score < 0.2:
                    evidence_against.append(f"low_anomaly_score:{anomaly_ev.final_anomaly_score:.2f}")

            out_edges = outgoing_edges.get(comp, [])
            if out_edges:
                avg_weight = sum(e.final_edge_weight for e in out_edges) / len(out_edges)
                if avg_weight > 0.3:
                    evidence_for.append(f"strong_outgoing_propagation:{avg_weight:.2f}")

            in_edges = incoming_edges.get(comp, [])
            if in_edges:
                avg_weight = sum(e.final_edge_weight for e in in_edges) / len(in_edges)
                if avg_weight > 0.5:
                    evidence_against.append(f"strong_incoming_influence:{avg_weight:.2f}")

            self.belief_manager.update_with_evidence(comp, evidence_for, evidence_against)

        self._log_step("EVIDENCE_COLLECT", "Collected evidence for hypotheses", {
            "state_summary": self.belief_manager.get_state_summary(),
        })

    def _spectral_validate(
        self,
        anomaly_evidence: List[MetricAnomalyEvidence],
    ) -> None:
        """Validate hypotheses using spectral anomaly types as belief states."""
        node_anomaly = {e.node_id: e for e in anomaly_evidence}

        for comp, hypothesis in self.belief_manager._hypotheses.items():
            anomaly_ev = node_anomaly.get(comp)
            if anomaly_ev and anomaly_ev.anomaly_type:
                self.belief_manager.confirm_with_spectral(comp, anomaly_ev.anomaly_type)

        self._log_step("SPECTRAL_VALIDATE", "Validated hypotheses with spectral evidence", {
            "state_summary": self.belief_manager.get_state_summary(),
        })

    def _abductive_evaluate(self) -> None:
        """Evaluate hypotheses and select the best explanation."""
        for comp, hypothesis in self.belief_manager._hypotheses.items():
            if hypothesis.belief_state in (BeliefState.SPECTRAL_CONFIRMED, BeliefState.EVIDENCE_SUPPORTED):
                if hypothesis.confidence > 0.6:
                    self.belief_manager.validate_hypothesis(comp)

        self._log_step("ABDUCTIVE_REASON", "Evaluated hypotheses", {
            "state_summary": self.belief_manager.get_state_summary(),
        })

    def _backtrack_low_confidence(self) -> None:
        """Backtrack from hypotheses with low confidence."""
        to_backtrack = []
        for comp, hypothesis in self.belief_manager._hypotheses.items():
            if hypothesis.belief_state in (
                BeliefState.EVIDENCE_CONTRADICTED,
                BeliefState.SPECTRAL_REJECTED,
            ):
                to_backtrack.append(comp)

        for comp in to_backtrack:
            self.belief_manager.backtrack(comp)

        self._log_step("BACKTRACK", f"Backtracked {len(to_backtrack)} hypotheses", {
            "backtracked_components": to_backtrack,
        })

    def _synthesize(
        self,
        original_ranking: List[RootCauseCandidate],
    ) -> List[RootCauseCandidate]:
        """Synthesize final root cause ranking from abductive reasoning results."""
        validated = self.belief_manager.get_hypotheses_by_state(BeliefState.VALIDATED)
        spectral_confirmed = self.belief_manager.get_hypotheses_by_state(BeliefState.SPECTRAL_CONFIRMED)
        evidence_supported = self.belief_manager.get_hypotheses_by_state(BeliefState.EVIDENCE_SUPPORTED)

        hypothesis_scores: Dict[str, float] = {}
        for h in validated:
            hypothesis_scores[h.component] = h.confidence
        for h in spectral_confirmed:
            hypothesis_scores[h.component] = max(
                hypothesis_scores.get(h.component, 0.0), h.confidence
            )
        for h in evidence_supported:
            hypothesis_scores[h.component] = max(
                hypothesis_scores.get(h.component, 0.0), h.confidence
            )

        refined: List[RootCauseCandidate] = []
        for cand in original_ranking:
            abductive_boost = hypothesis_scores.get(cand.node_id, 0.0)
            new_score = cand.root_score * (1.0 + abductive_boost)

            spectral_type = None
            for h in validated + spectral_confirmed:
                if h.component == cand.node_id:
                    spectral_type = h.spectral_anomaly_type
                    break

            explanation = cand.explanation
            if spectral_type and spectral_type != "no_strong_spectral_anomaly":
                explanation += f" [Spectral: {spectral_type}]"
            if cand.node_id in hypothesis_scores:
                explanation += f" [Abductive: {hypothesis_scores[cand.node_id]:.2f}]"

            refined.append(RootCauseCandidate(
                node_id=cand.node_id,
                root_score=round(new_score, 4),
                anomaly_score=cand.anomaly_score,
                out_evidence=cand.out_evidence,
                in_evidence=cand.in_evidence,
                onset_time=cand.onset_time,
                explanation=explanation,
            ))

        refined.sort(key=lambda c: c.root_score, reverse=True)
        return refined

    def _infer_reason_from_anomaly_type(self, anomaly_type: str) -> str:
        """Infer a likely root cause reason from spectral anomaly type."""
        mapping = {
            "slow_trend_anomaly": "resource_exhaustion",
            "fast_burst_or_jitter_anomaly": "sudden_spike_or_instability",
            "periodic_oscillation_anomaly": "periodic_interference_or_misconfiguration",
            "mixed_spectral_anomaly": "complex_failure",
            "no_strong_spectral_anomaly": "unknown",
        }
        return mapping.get(anomaly_type, "unknown")

    def _log_step(self, step: str, description: str, details: Dict[str, Any]) -> None:
        """Log a reasoning step to the trajectory."""
        self._trajectory.append({
            "step": step,
            "description": description,
            "iteration": self.belief_manager._iteration,
            "details": details,
        })
