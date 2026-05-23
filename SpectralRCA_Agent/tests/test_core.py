"""
Unit tests for SpectralRCA-Agent core modules.
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.anomaly.traditional import (
    robust_median_mad,
    robust_z_scores,
    threshold_score,
    change_rate_score,
    max_deviation_score,
    traditional_anomaly_score,
)
from src.anomaly.spectral import (
    center_signal,
    compute_fft_power,
    compute_spectral_features,
    split_baseline_windows,
    compute_baseline_spectral_stats,
    spectral_anomaly_score,
    classify_spectral_anomaly,
)
from src.anomaly.metric_anomaly_expert import MetricAnomalyExpert
from src.config import AnomalyConfig, SpectralRCAConfig
from src.graph.spectral_edge import (
    compute_cross_correlation,
    compute_spectral_shape_similarity,
    compute_dominant_freq_match,
    compute_spectral_edge_score,
)
from src.graph.graph_consistency import compute_graph_consistency
from src.graph.graph_refiner import SpectralGraphRefinementExpert
from src.ranking.root_cause_ranker import RootCauseRanker
from src.reasoning.belief_state import BeliefStateManager
from src.reasoning.abductive_engine import AbductiveReasoningEngine
from src.schemas import (
    BeliefState,
    MetricAnomalyEvidence,
    EdgeEvidence,
    RootCauseCandidate,
    SpectralExperience,
)


class TestTraditionalAnomaly(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.baseline = np.random.randn(100) * 10 + 50
        self.incident_normal = np.random.randn(30) * 10 + 50
        self.incident_anomalous = np.random.randn(30) * 10 + 80

    def test_robust_median_mad(self):
        median, mad = robust_median_mad(self.baseline)
        self.assertAlmostEqual(median, np.median(self.baseline), places=5)
        self.assertGreater(mad, 0)

    def test_robust_z_scores_normal(self):
        z = robust_z_scores(self.incident_normal, self.baseline)
        self.assertEqual(len(z), len(self.incident_normal))
        self.assertLess(np.max(np.abs(z)), 5.0)

    def test_robust_z_scores_anomalous(self):
        z = robust_z_scores(self.incident_anomalous, self.baseline)
        self.assertGreater(np.max(np.abs(z)), 3.0)

    def test_traditional_anomaly_score(self):
        score_normal, _, _, _, _ = traditional_anomaly_score(self.incident_normal, self.baseline)
        score_anomalous, _, _, _, _ = traditional_anomaly_score(self.incident_anomalous, self.baseline)
        self.assertGreater(score_anomalous, score_normal)


class TestSpectralAnomaly(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.normal_signal = np.random.randn(60) * 5 + 100
        self.trend_signal = np.concatenate([
            np.random.randn(20) * 2 + 100,
            np.linspace(100, 150, 20),
            np.random.randn(20) * 2 + 150,
        ])
        self.burst_signal = np.concatenate([
            np.random.randn(25) * 2 + 100,
            np.random.randn(10) * 50 + 100,
            np.random.randn(25) * 2 + 100,
        ])

    def test_center_signal(self):
        centered = center_signal(self.normal_signal)
        self.assertAlmostEqual(np.mean(centered), 0.0, places=10)

    def test_compute_fft_power(self):
        X, power, freqs = compute_fft_power(self.normal_signal)
        self.assertEqual(len(power), len(freqs))
        self.assertGreater(np.sum(power), 0)

    def test_compute_spectral_features(self):
        features = compute_spectral_features(self.normal_signal)
        self.assertGreater(features.total_energy, 0)
        self.assertAlmostEqual(features.low_ratio + features.mid_ratio + features.high_ratio, 1.0, places=5)
        self.assertGreater(features.spectral_entropy, 0)

    def test_spectral_anomaly_score(self):
        score_normal, _, _, _ = spectral_anomaly_score(self.normal_signal, self.normal_signal)
        score_anomalous, _, _, _ = spectral_anomaly_score(self.trend_signal, self.normal_signal)
        self.assertGreater(score_anomalous, score_normal)

    def test_classify_spectral_anomaly(self):
        features = compute_spectral_features(self.trend_signal)
        _, _, z_values, _ = spectral_anomaly_score(self.trend_signal, self.normal_signal)
        anomaly_type = classify_spectral_anomaly(features, z_values)
        self.assertIn(anomaly_type, [
            "slow_trend_anomaly", "fast_burst_or_jitter_anomaly",
            "periodic_oscillation_anomaly", "mixed_spectral_anomaly",
            "no_strong_spectral_anomaly",
        ])


class TestMetricAnomalyExpert(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.config = SpectralRCAConfig()
        self.expert = MetricAnomalyExpert(self.config)

    def test_detect_single_series(self):
        baseline = np.random.randn(90) * 5 + 100
        incident = np.concatenate([
            np.random.randn(10) * 5 + 100,
            np.random.randn(10) * 5 + 150,
            np.random.randn(10) * 5 + 100,
        ])
        evidence = self.expert.detect_single_series("test.node", incident, baseline)
        self.assertEqual(evidence.node_id, "test.node")
        self.assertGreaterEqual(evidence.final_anomaly_score, 0.0)
        self.assertLessEqual(evidence.final_anomaly_score, 1.0)

    def test_detect_single_series_normal(self):
        baseline = np.random.randn(90) * 5 + 100
        incident = np.random.randn(30) * 5 + 100
        evidence = self.expert.detect_single_series("test.normal", incident, baseline)
        self.assertLess(evidence.final_anomaly_score, 0.5)


class TestSpectralEdge(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.source = np.random.randn(60) * 5 + 100
        self.target_lagged = np.concatenate([np.zeros(5), self.source[:-5]]) + np.random.randn(60) * 0.5
        self.target_independent = np.random.randn(60) * 5 + 200

    def test_cross_correlation(self):
        corr, lag = compute_cross_correlation(self.source, self.target_lagged, max_lag=10)
        self.assertGreater(abs(corr), 0.1)

    def test_spectral_shape_similarity(self):
        sim_lagged = compute_spectral_shape_similarity(self.source, self.target_lagged)
        sim_independent = compute_spectral_shape_similarity(self.source, self.target_independent)
        self.assertGreater(sim_lagged, sim_independent)

    def test_spectral_edge_score(self):
        score, shape, freq, phase = compute_spectral_edge_score(
            self.source, self.target_lagged, best_lag=5,
        )
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


class TestGraphConsistency(unittest.TestCase):
    def test_graph_consistency(self):
        edge_weights = {("A", "B"): 0.8, ("B", "C"): 0.5}
        node_scores = {"A": 0.9, "B": 0.6, "C": 0.3}
        consistency = compute_graph_consistency(edge_weights, node_scores)
        self.assertIn(("A", "B"), consistency)
        self.assertGreater(consistency[("A", "B")], 0.5)


class TestRootCauseRanker(unittest.TestCase):
    def test_rank(self):
        anomaly_evidence = [
            MetricAnomalyEvidence(
                node_id="A", traditional_score=0.8, spectral_score=0.7,
                final_anomaly_score=0.75, anomaly_type="slow_trend_anomaly",
                max_abs_robust_z=4.5, max_change_rate=0.3, max_deviation_score=0.6,
                total_energy=1000.0, spectral_energy_z=3.5, low_ratio=0.6,
                mid_ratio=0.3, high_ratio=0.1, dominant_freq_index=1,
                dominant_freq_energy_ratio=0.5, spectral_entropy=2.0,
                quality_flag="ok", explanation="test A",
            ),
            MetricAnomalyEvidence(
                node_id="B", traditional_score=0.3, spectral_score=0.2,
                final_anomaly_score=0.25, anomaly_type="no_strong_spectral_anomaly",
                max_abs_robust_z=1.5, max_change_rate=0.1, max_deviation_score=0.2,
                total_energy=500.0, spectral_energy_z=1.0, low_ratio=0.4,
                mid_ratio=0.3, high_ratio=0.3, dominant_freq_index=3,
                dominant_freq_energy_ratio=0.2, spectral_entropy=3.0,
                quality_flag="ok", explanation="test B",
            ),
        ]
        edge_evidence = [
            EdgeEvidence(
                source="A", target="B", prior_weight=0.5,
                node_anomaly_factor=1.75, best_lag=3, lag_corr=0.7,
                spectral_shape_similarity=0.8, dominant_freq_match=1.0,
                phase_lag_consistency=0.6, graph_consistency=0.9,
                final_edge_weight=0.8, keep_edge=True,
                explanation="test edge",
            ),
        ]
        ranker = RootCauseRanker()
        ranked = ranker.rank(anomaly_evidence, edge_evidence)
        self.assertEqual(len(ranked), 2)
        self.assertEqual(ranked[0].node_id, "A")
        self.assertGreater(ranked[0].root_score, ranked[1].root_score)


class TestBeliefState(unittest.TestCase):
    def test_create_and_update(self):
        manager = BeliefStateManager()
        h = manager.create_hypothesis("comp_A", reason="resource_exhaustion", spectral_anomaly_type="slow_trend_anomaly")
        self.assertEqual(h.belief_state, BeliefState.HYPOTHESIZED)

        h = manager.update_with_evidence("comp_A", evidence_for=["high_anomaly"])
        self.assertEqual(h.belief_state, BeliefState.EVIDENCE_SUPPORTED)

        h = manager.confirm_with_spectral("comp_A", "slow_trend_anomaly")
        self.assertEqual(h.belief_state, BeliefState.SPECTRAL_CONFIRMED)

        h = manager.validate_hypothesis("comp_A")
        self.assertEqual(h.belief_state, BeliefState.VALIDATED)

    def test_backtrack(self):
        manager = BeliefStateManager()
        manager.create_hypothesis("comp_B")
        manager.update_with_evidence("comp_B", evidence_against=["low_anomaly", "contradicted"])
        self.assertEqual(manager._hypotheses["comp_B"].belief_state, BeliefState.EVIDENCE_CONTRADICTED)

        manager.backtrack("comp_B")
        self.assertEqual(manager._hypotheses["comp_B"].belief_state, BeliefState.HYPOTHESIZED)


class TestExperienceStore(unittest.TestCase):
    def test_add_and_retrieve(self):
        from src.memory.experience_store import ExperienceStore
        from src.config import MemoryConfig

        config = MemoryConfig(frozen=False, memory_dir=None)
        store = ExperienceStore(config)

        exp = SpectralExperience(
            component="Redis02", kpi="memory_usage",
            anomaly_type="slow_trend_anomaly",
            spectral_pattern={"total_energy": 1000.0, "low_ratio": 0.7},
            reason="high memory usage", success=True,
            timestamp="2021-03-04 14:57:00", case_id="test_1",
        )
        store.add(exp)

        retrieved = store.retrieve_by_component("Redis02")
        self.assertEqual(len(retrieved), 1)
        self.assertEqual(retrieved[0].component, "Redis02")

    def test_retrieve_similar(self):
        from src.memory.experience_store import ExperienceStore
        from src.config import MemoryConfig

        config = MemoryConfig(frozen=False, memory_dir=None)
        store = ExperienceStore(config)

        store.add(SpectralExperience(
            component="Redis02", kpi="memory_usage",
            anomaly_type="slow_trend_anomaly",
            spectral_pattern={"total_energy": 1000.0, "low_ratio": 0.7, "high_ratio": 0.1},
            reason="high memory usage", success=True,
            timestamp="2021-03-04 14:57:00", case_id="test_1",
        ))
        store.add(SpectralExperience(
            component="Tomcat01", kpi="cpu_usage",
            anomaly_type="fast_burst_or_jitter_anomaly",
            spectral_pattern={"total_energy": 800.0, "low_ratio": 0.2, "high_ratio": 0.6},
            reason="CPU fault", success=True,
            timestamp="2021-03-06 06:20:00", case_id="test_2",
        ))

        results = store.retrieve_similar(
            "Redis02", "slow_trend_anomaly",
            {"total_energy": 950.0, "low_ratio": 0.65, "high_ratio": 0.15},
        )
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0][0].component, "Redis02")


if __name__ == "__main__":
    unittest.main()
