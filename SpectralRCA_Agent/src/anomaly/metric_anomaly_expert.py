from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.anomaly.traditional import robust_median_mad, robust_z_scores, traditional_anomaly_score
from src.anomaly.spectral import (
    compute_spectral_features,
    spectral_anomaly_score,
    classify_spectral_anomaly,
)
from src.config import AnomalyConfig, SpectralRCAConfig
from src.schemas import MetricAnomalyEvidence


class MetricAnomalyExpert:
    """Spectral-enhanced metric anomaly detection expert.

    Combines traditional anomaly detection with frequency-domain analysis
    to provide comprehensive anomaly evidence for each component.metric pair.
    """

    def __init__(self, config: Optional[SpectralRCAConfig] = None) -> None:
        self.config = config or SpectralRCAConfig()
        self.anomaly_cfg = self.config.anomaly

    def detect(
        self,
        metric_series_dict: Dict[str, pd.DataFrame],
        incident_start: str,
        incident_end: str,
        baseline_start: str,
        baseline_end: str,
    ) -> List[MetricAnomalyEvidence]:
        """Run anomaly detection on all component.metric series.

        Args:
            metric_series_dict: Mapping from node_id to DataFrame with
                columns [timestamp, value] already resampled.
            incident_start/end: Time strings for incident window.
            baseline_start/end: Time strings for baseline window.

        Returns:
            List of MetricAnomalyEvidence for each component.metric.
        """
        results: List[MetricAnomalyEvidence] = []
        for node_id, df in metric_series_dict.items():
            evidence = self._detect_single(node_id, df, incident_start, incident_end, baseline_start, baseline_end)
            if evidence is not None:
                results.append(evidence)
        return results

    def detect_single_series(
        self,
        node_id: str,
        incident_series: np.ndarray,
        baseline_series: np.ndarray,
        quality_flag: str = "ok",
    ) -> MetricAnomalyEvidence:
        """Run anomaly detection on a single pre-extracted series pair.

        This method is useful for standalone evaluation and ablation.
        """
        return self._compute_evidence(node_id, incident_series, baseline_series, quality_flag)

    def _detect_single(
        self,
        node_id: str,
        df: pd.DataFrame,
        incident_start: str,
        incident_end: str,
        baseline_start: str,
        baseline_end: str,
    ) -> Optional[MetricAnomalyEvidence]:
        """Detect anomalies for a single component.metric."""
        if "timestamp" not in df.columns or "value" not in df.columns:
            return None
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").set_index("timestamp")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

        missing_ratio = df["value"].isna().sum() / max(len(df), 1)
        quality_flag = "ok"
        if missing_ratio > self.config.pipeline.missing_threshold:
            quality_flag = "low_quality"

        df["value"] = df["value"].interpolate(method="linear")
        df["value"] = df["value"].ffill().bfill()

        baseline_mask = (df.index >= pd.to_datetime(baseline_start)) & (df.index <= pd.to_datetime(baseline_end))
        incident_mask = (df.index >= pd.to_datetime(incident_start)) & (df.index <= pd.to_datetime(incident_end))

        baseline_series = df.loc[baseline_mask, "value"].dropna().values
        incident_series = df.loc[incident_mask, "value"].dropna().values

        if len(baseline_series) < 4 or len(incident_series) < 2:
            return self._make_empty_evidence(node_id, "insufficient_data")

        if len(incident_series) < self.config.pipeline.min_spectral_length:
            quality_flag = "too_short_for_spectral"

        return self._compute_evidence(node_id, incident_series, baseline_series, quality_flag)

    def _compute_evidence(
        self,
        node_id: str,
        incident_series: np.ndarray,
        baseline_series: np.ndarray,
        quality_flag: str,
    ) -> MetricAnomalyEvidence:
        """Compute full anomaly evidence for a single series."""
        cfg = self.anomaly_cfg

        traditional_score = 0.0
        thr_sc = zscore_sc = cr_sc = dev_sc = 0.0
        max_abs_robust_z = 0.0
        max_change_rate = 0.0
        max_deviation = 0.0

        if self.config.enable_traditional_anomaly:
            traditional_score, thr_sc, zscore_sc, cr_sc, dev_sc = traditional_anomaly_score(
                incident_series, baseline_series, config=cfg,
            )
            z_scores = robust_z_scores(incident_series, baseline_series, cfg)
            max_abs_robust_z = float(np.max(np.abs(z_scores)))
            if len(incident_series) >= 2:
                rates = np.abs(np.diff(incident_series)) / (np.abs(incident_series[:-1]) + cfg.eps)
                max_change_rate = float(np.max(rates))
            median, _ = robust_median_mad(baseline_series)
            max_deviation = float(np.max(np.abs(incident_series - median)))

        spectral_score = 0.0
        total_energy = 0.0
        spectral_energy_z = 0.0
        low_ratio = 0.0
        mid_ratio = 0.0
        high_ratio = 0.0
        dominant_freq_index = 0
        dominant_freq_energy_ratio = 0.0
        spectral_entropy = 0.0
        anomaly_type = "no_strong_spectral_anomaly"

        if self.config.enable_spectral_anomaly and quality_flag != "too_short_for_spectral":
            spectral_score, features, z_values, _ = spectral_anomaly_score(
                incident_series, baseline_series, cfg,
            )
            total_energy = features.total_energy
            spectral_energy_z = z_values.get("spectral_energy_z", 0.0)
            low_ratio = features.low_ratio
            mid_ratio = features.mid_ratio
            high_ratio = features.high_ratio
            dominant_freq_index = features.dominant_freq_index
            dominant_freq_energy_ratio = features.dominant_freq_energy_ratio
            spectral_entropy = features.spectral_entropy
            anomaly_type = classify_spectral_anomaly(features, z_values, cfg)

        final_score = cfg.final_weight_traditional * traditional_score + cfg.final_weight_spectral * spectral_score

        if final_score >= cfg.high_confidence_threshold:
            level = "high_confidence_anomaly"
        elif final_score >= cfg.candidate_threshold:
            level = "candidate_anomaly"
        else:
            level = "normal_or_weak"

        explanation = self._build_explanation(
            node_id, level, anomaly_type, traditional_score, spectral_score,
            max_abs_robust_z, max_change_rate, total_energy,
        )

        return MetricAnomalyEvidence(
            node_id=node_id,
            traditional_score=round(traditional_score, 4),
            spectral_score=round(spectral_score, 4),
            final_anomaly_score=round(final_score, 4),
            anomaly_type=anomaly_type,
            max_abs_robust_z=round(max_abs_robust_z, 4),
            max_change_rate=round(max_change_rate, 4),
            max_deviation_score=round(dev_sc, 4),
            total_energy=round(total_energy, 4),
            spectral_energy_z=round(spectral_energy_z, 4),
            low_ratio=round(low_ratio, 4),
            mid_ratio=round(mid_ratio, 4),
            high_ratio=round(high_ratio, 4),
            dominant_freq_index=dominant_freq_index,
            dominant_freq_energy_ratio=round(dominant_freq_energy_ratio, 4),
            spectral_entropy=round(spectral_entropy, 4),
            quality_flag=quality_flag,
            explanation=explanation,
        )

    def _build_explanation(
        self,
        node_id: str,
        level: str,
        anomaly_type: str,
        traditional_score: float,
        spectral_score: float,
        max_z: float,
        max_cr: float,
        total_energy: float,
    ) -> str:
        """Build human-readable explanation for anomaly evidence."""
        parts = [f"{node_id}"]
        if level == "high_confidence_anomaly":
            parts.append("shows high-confidence anomaly")
        elif level == "candidate_anomaly":
            parts.append("shows candidate anomaly")
        else:
            parts.append("appears normal or weak")

        if traditional_score > 0.5:
            parts.append(f"high traditional score ({traditional_score:.2f})")
        if max_z > 3.0:
            parts.append(f"elevated robust z-score ({max_z:.2f})")
        if spectral_score > 0.5:
            parts.append(f"high spectral score ({spectral_score:.2f})")
        if anomaly_type == "slow_trend_anomaly":
            parts.append("suggesting slow trend anomaly")
        elif anomaly_type == "fast_burst_or_jitter_anomaly":
            parts.append("suggesting burst/jitter anomaly")
        elif anomaly_type == "periodic_oscillation_anomaly":
            parts.append("suggesting periodic oscillation")
        elif anomaly_type == "mixed_spectral_anomaly":
            parts.append("suggesting mixed spectral anomaly")

        return ", ".join(parts) + "."

    def _make_empty_evidence(self, node_id: str, quality_flag: str) -> MetricAnomalyEvidence:
        """Create an empty evidence object for insufficient data cases."""
        return MetricAnomalyEvidence(
            node_id=node_id, traditional_score=0.0, spectral_score=0.0,
            final_anomaly_score=0.0, anomaly_type="no_strong_spectral_anomaly",
            max_abs_robust_z=0.0, max_change_rate=0.0, max_deviation_score=0.0,
            total_energy=0.0, spectral_energy_z=0.0, low_ratio=0.0, mid_ratio=0.0,
            high_ratio=0.0, dominant_freq_index=0, dominant_freq_energy_ratio=0.0,
            spectral_entropy=0.0, quality_flag=quality_flag,
            explanation=f"{node_id} has insufficient data for anomaly detection.",
        )
