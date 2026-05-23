from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from src.config import GraphConfig


def compute_graph_consistency(
    edge_weights: Dict[Tuple[str, str], float],
    node_anomaly_scores: Dict[str, float],
    config: Optional[GraphConfig] = None,
) -> Dict[Tuple[str, str], float]:
    """Compute graph structure consistency for each edge.

    Graph consistency measures whether an edge's weight is consistent
    with the anomaly scores of its connected nodes. An edge from a
    highly anomalous source to a less anomalous target is more likely
    to represent true causal propagation.

    Args:
        edge_weights: Dict mapping (source, target) to edge weight.
        node_anomaly_scores: Dict mapping node_id to anomaly score.
        config: GraphConfig instance.

    Returns:
        Dict mapping (source, target) to consistency score in [0, 1].
    """
    cfg = config or GraphConfig()
    eta = cfg.graph_consistency_eta

    consistency: Dict[Tuple[str, str], float] = {}

    for (src, tgt), weight in edge_weights.items():
        src_score = node_anomaly_scores.get(src, 0.0)
        tgt_score = node_anomaly_scores.get(tgt, 0.0)

        if src_score < 1e-8 and tgt_score < 1e-8:
            consistency[(src, tgt)] = 0.0
            continue

        score_ratio = src_score / (src_score + tgt_score + 1e-12)

        if score_ratio >= 0.5:
            consistency[(src, tgt)] = 1.0 - eta * (1.0 - score_ratio)
        else:
            consistency[(src, tgt)] = max(0.0, score_ratio / (eta + 1e-12))

    return consistency


def compute_propagation_path_score(
    path: List[str],
    edge_weights: Dict[Tuple[str, str], float],
    node_anomaly_scores: Dict[str, float],
) -> float:
    """Compute the overall propagation consistency score for a causal path.

    A valid propagation path should have monotonically decreasing
    anomaly scores from source to sink.

    Returns:
        Path consistency score in [0, 1].
    """
    if len(path) < 2:
        return 0.0

    edge_scores = []
    for i in range(len(path) - 1):
        key = (path[i], path[i + 1])
        edge_scores.append(edge_weights.get(key, 0.0))

    if not edge_scores:
        return 0.0

    avg_edge = np.mean(edge_scores)

    node_scores = [node_anomaly_scores.get(n, 0.0) for n in path]
    monotonic_penalty = 0.0
    for i in range(len(node_scores) - 1):
        if node_scores[i] < node_scores[i + 1]:
            monotonic_penalty += (node_scores[i + 1] - node_scores[i])

    max_possible_penalty = max(sum(node_scores), 1e-12)
    monotonic_score = max(0.0, 1.0 - monotonic_penalty / max_possible_penalty)

    return float(avg_edge * monotonic_score)
