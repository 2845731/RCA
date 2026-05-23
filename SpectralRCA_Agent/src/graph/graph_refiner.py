from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.anomaly.spectral import compute_spectral_features
from src.config import GraphConfig, SpectralRCAConfig
from src.graph.spectral_edge import (
    compute_cross_correlation,
    compute_spectral_edge_score,
)
from src.graph.graph_consistency import compute_graph_consistency
from src.schemas import EdgeEvidence, MetricAnomalyEvidence


class SpectralGraphRefinementExpert:
    """Spectral-validated causal graph refinement expert.

    Validates and refines causal edges using frequency-domain evidence:
    1. Cross-correlation lag analysis
    2. Spectral shape similarity
    3. Dominant frequency matching
    4. Phase lag consistency
    5. Graph structure consistency

    This implements the core of Innovation 2: Spectral-Validated Causal Graph.
    """

    def __init__(self, config: Optional[SpectralRCAConfig] = None) -> None:
        self.config = config or SpectralRCAConfig()
        self.graph_cfg = self.config.graph

    def refine(
        self,
        anomaly_evidence: List[MetricAnomalyEvidence],
        metric_series_dict: Dict[str, pd.DataFrame],
        prior_edges: Optional[List[Tuple[str, str, float]]] = None,
        incident_start: Optional[str] = None,
        incident_end: Optional[str] = None,
    ) -> List[EdgeEvidence]:
        """Refine causal edges using spectral validation.

        Args:
            anomaly_evidence: Anomaly detection results for each node.
            metric_series_dict: Mapping from node_id to DataFrame with
                [timestamp, value] columns.
            prior_edges: Optional list of (source, target, prior_weight) tuples.
                If None, all pairwise edges are considered.
            incident_start/end: Time window for extracting incident series.

        Returns:
            List of EdgeEvidence for each validated edge.
        """
        node_scores = {e.node_id: e.final_anomaly_score for e in anomaly_evidence}
        node_series = self._extract_incident_series(metric_series_dict, incident_start, incident_end)

        if prior_edges is None:
            prior_edges = self._generate_candidate_edges(anomaly_evidence)

        edge_weights_raw: Dict[Tuple[str, str], float] = {}
        results: List[EdgeEvidence] = []

        for source, target, prior_weight in prior_edges:
            evidence = self._validate_edge(
                source, target, prior_weight,
                node_scores, node_series,
            )
            if evidence is not None:
                results.append(evidence)
                edge_weights_raw[(source, target)] = evidence.final_edge_weight

        if self.config.enable_graph_consistency and edge_weights_raw:
            consistency_scores = compute_graph_consistency(
                edge_weights_raw, node_scores, self.graph_cfg,
            )
            for evidence in results:
                key = (evidence.source, evidence.target)
                gc = consistency_scores.get(key, 0.0)
                evidence.graph_consistency = round(gc, 4)
                evidence.final_edge_weight = round(
                    evidence.final_edge_weight * gc, 4,
                )
                evidence.keep_edge = evidence.final_edge_weight >= self.graph_cfg.pruning_threshold
                evidence.explanation = self._update_explanation(evidence)

        return results

    def _validate_edge(
        self,
        source: str,
        target: str,
        prior_weight: float,
        node_scores: Dict[str, float],
        node_series: Dict[str, np.ndarray],
    ) -> Optional[EdgeEvidence]:
        """Validate a single edge using spectral and temporal evidence."""
        src_series = node_series.get(source)
        tgt_series = node_series.get(target)

        if src_series is None or tgt_series is None:
            return None
        if len(src_series) < 4 or len(tgt_series) < 4:
            return None

        max_lag = self.graph_cfg.max_lag or max(1, min(len(src_series), len(tgt_series)) // 4)

        lag_corr, best_lag = 0.0, 0
        if self.config.enable_lag_corr:
            lag_corr, best_lag = compute_cross_correlation(src_series, tgt_series, max_lag)

        spectral_score = 0.0
        shape_sim = 0.0
        freq_match = 0.0
        phase_cons = 0.0

        if self.config.enable_spectral_shape or self.config.enable_dominant_freq_match or self.config.enable_phase_lag:
            spectral_score, shape_sim, freq_match, phase_cons = compute_spectral_edge_score(
                src_series, tgt_series, best_lag, self.graph_cfg,
            )

        src_anomaly = node_scores.get(source, 0.0)
        node_anomaly_factor = 1.0 + src_anomaly

        normalized_lag_corr = max(0.0, lag_corr)

        final_weight = (
            prior_weight
            * node_anomaly_factor
            * normalized_lag_corr
            * (1.0 + spectral_score)
        )

        keep = final_weight >= self.graph_cfg.pruning_threshold

        explanation = (
            f"Edge {source}→{target}: prior={prior_weight:.3f}, "
            f"lag_corr={lag_corr:.3f}(lag={best_lag}), "
            f"spectral={spectral_score:.3f}(shape={shape_sim:.3f},"
            f"freq={freq_match:.3f},phase={phase_cons:.3f}), "
            f"final={final_weight:.3f}, keep={keep}"
        )

        return EdgeEvidence(
            source=source,
            target=target,
            prior_weight=round(prior_weight, 4),
            node_anomaly_factor=round(node_anomaly_factor, 4),
            best_lag=best_lag,
            lag_corr=round(lag_corr, 4),
            spectral_shape_similarity=round(shape_sim, 4),
            dominant_freq_match=round(freq_match, 4),
            phase_lag_consistency=round(phase_cons, 4),
            graph_consistency=0.0,
            final_edge_weight=round(final_weight, 4),
            keep_edge=keep,
            explanation=explanation,
        )

    def _extract_incident_series(
        self,
        metric_series_dict: Dict[str, pd.DataFrame],
        incident_start: Optional[str],
        incident_end: Optional[str],
    ) -> Dict[str, np.ndarray]:
        """Extract incident window series for each node."""
        result: Dict[str, np.ndarray] = {}
        for node_id, df in metric_series_dict.items():
            if "timestamp" not in df.columns or "value" not in df.columns:
                continue
            df = df.copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")

            if incident_start and incident_end:
                mask = (df["timestamp"] >= pd.to_datetime(incident_start)) & (
                    df["timestamp"] <= pd.to_datetime(incident_end)
                )
                values = df.loc[mask, "value"].dropna().values
            else:
                values = df["value"].dropna().values

            if len(values) >= 4:
                result[node_id] = values
        return result

    def _generate_candidate_edges(
        self,
        anomaly_evidence: List[MetricAnomalyEvidence],
    ) -> List[Tuple[str, str, float]]:
        """Generate candidate edges from anomaly evidence.

        Creates edges from highly anomalous nodes to moderately anomalous nodes,
        with prior weights based on anomaly score differences.
        Only considers top-K most anomalous nodes to avoid O(n^2) explosion.
        """
        candidates: List[Tuple[str, str, float]] = []
        sorted_evidence = sorted(anomaly_evidence, key=lambda e: e.final_anomaly_score, reverse=True)

        top_sources = [e for e in sorted_evidence if e.final_anomaly_score >= 0.5][:20]
        top_targets = [e for e in sorted_evidence if e.final_anomaly_score >= 0.3][:30]

        for i, src in enumerate(top_sources):
            for j, tgt in enumerate(top_targets):
                if src.node_id == tgt.node_id:
                    continue
                prior = 0.5 * (src.final_anomaly_score - tgt.final_anomaly_score + 0.5)
                prior = max(0.1, min(1.0, prior))
                candidates.append((src.node_id, tgt.node_id, prior))

        return candidates

    def _update_explanation(self, evidence: EdgeEvidence) -> str:
        """Update explanation with graph consistency info."""
        base = evidence.explanation
        gc = evidence.graph_consistency
        return f"{base}, graph_consistency={gc:.3f}"
