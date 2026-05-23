from __future__ import annotations

from typing import Dict, List, Optional

from src.schemas import EdgeEvidence, MetricAnomalyEvidence, RootCauseCandidate


class RootCauseRanker:
    """Rank root cause candidates by integrating anomaly, graph, and temporal evidence.

    Scoring formula:
        root_score = alpha * anomaly_score
                   + beta * out_evidence
                   - gamma * in_evidence
                   + delta * onset_bonus

    where:
        - anomaly_score: node's final anomaly score
        - out_evidence: sum of outgoing edge weights (propagation strength)
        - in_evidence: sum of incoming edge weights (being affected by others)
        - onset_bonus: earlier onset time gets higher bonus
    """

    def __init__(
        self,
        alpha: float = 0.40,
        beta: float = 0.35,
        gamma: float = 0.20,
        delta: float = 0.05,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def rank(
        self,
        anomaly_evidence: List[MetricAnomalyEvidence],
        edge_evidence: List[EdgeEvidence],
        onset_times: Optional[Dict[str, str]] = None,
    ) -> List[RootCauseCandidate]:
        """Rank root cause candidates.

        Args:
            anomaly_evidence: Anomaly detection results.
            edge_evidence: Validated causal edges.
            onset_times: Optional mapping from node_id to onset time string.

        Returns:
            List of RootCauseCandidate sorted by root_score descending.
        """
        node_scores = {e.node_id: e.final_anomaly_score for e in anomaly_evidence}
        node_types = {e.node_id: e.anomaly_type for e in anomaly_evidence}

        out_evidence: Dict[str, float] = {}
        in_evidence: Dict[str, float] = {}

        for edge in edge_evidence:
            if not edge.keep_edge:
                continue
            out_evidence[edge.source] = out_evidence.get(edge.source, 0.0) + edge.final_edge_weight
            in_evidence[edge.target] = in_evidence.get(edge.target, 0.0) + edge.final_edge_weight

        all_nodes = set(node_scores.keys())
        onset_times = onset_times or {}

        min_onset_ts = float("inf")
        max_onset_ts = 0.0
        for node_id, t_str in onset_times.items():
            try:
                from datetime import datetime
                ts = datetime.strptime(t_str, "%Y-%m-%d %H:%M:%S").timestamp()
                min_onset_ts = min(min_onset_ts, ts)
                max_onset_ts = max(max_onset_ts, ts)
            except (ValueError, OSError):
                pass

        onset_range = max(max_onset_ts - min_onset_ts, 1.0)

        candidates: List[RootCauseCandidate] = []
        for node_id in all_nodes:
            anomaly_score = node_scores.get(node_id, 0.0)
            out_ev = out_evidence.get(node_id, 0.0)
            in_ev = in_evidence.get(node_id, 0.0)

            onset_bonus = 0.0
            if node_id in onset_times:
                try:
                    from datetime import datetime
                    ts = datetime.strptime(onset_times[node_id], "%Y-%m-%d %H:%M:%S").timestamp()
                    onset_bonus = 1.0 - (ts - min_onset_ts) / onset_range
                except (ValueError, OSError):
                    onset_bonus = 0.0

            root_score = (
                self.alpha * anomaly_score
                + self.beta * out_ev
                - self.gamma * in_ev
                + self.delta * onset_bonus
            )
            root_score = max(0.0, root_score)

            anomaly_type = node_types.get(node_id, "")
            explanation = self._build_explanation(
                node_id, anomaly_score, out_ev, in_ev, onset_bonus, anomaly_type,
            )

            candidates.append(RootCauseCandidate(
                node_id=node_id,
                root_score=round(root_score, 4),
                anomaly_score=round(anomaly_score, 4),
                out_evidence=round(out_ev, 4),
                in_evidence=round(in_ev, 4),
                onset_time=onset_times.get(node_id),
                explanation=explanation,
            ))

        candidates.sort(key=lambda c: c.root_score, reverse=True)
        return candidates

    def _build_explanation(
        self,
        node_id: str,
        anomaly_score: float,
        out_ev: float,
        in_ev: float,
        onset_bonus: float,
        anomaly_type: str,
    ) -> str:
        """Build explanation for a root cause candidate."""
        parts = [f"Node {node_id}"]
        if anomaly_score > 0.7:
            parts.append("has high anomaly score")
        elif anomaly_score > 0.4:
            parts.append("has moderate anomaly score")
        else:
            parts.append("has low anomaly score")

        if out_ev > 0.5:
            parts.append("strong outgoing propagation")
        if in_ev > 0.3:
            parts.append("significantly affected by others")
        if onset_bonus > 0.5:
            parts.append("early onset time")
        if anomaly_type and anomaly_type != "no_strong_spectral_anomaly":
            parts.append(f"spectral type: {anomaly_type}")

        return ", ".join(parts) + "."
