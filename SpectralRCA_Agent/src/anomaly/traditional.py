from __future__ import annotations

import numpy as np
from typing import Optional, Tuple

from src.config import AnomalyConfig


def robust_median_mad(series: np.ndarray) -> Tuple[float, float]:
    """Compute median and Median Absolute Deviation (MAD) of a series."""
    median = float(np.median(series))
    mad = float(np.median(np.abs(series - median)))
    return median, mad


def robust_z_scores(
    incident: np.ndarray,
    baseline: np.ndarray,
    config: Optional[AnomalyConfig] = None,
) -> np.ndarray:
    """Compute robust z-scores for incident series against baseline distribution.

    Uses formula: robust_z = (x - median(baseline)) / (1.4826 * MAD(baseline) + eps)
    """
    cfg = config or AnomalyConfig()
    median, mad = robust_median_mad(baseline)
    scale = 1.4826 * mad + cfg.eps
    return (incident - median) / scale


def threshold_score(
    incident: np.ndarray,
    baseline: np.ndarray,
    optional_threshold: Optional[float] = None,
    config: Optional[AnomalyConfig] = None,
) -> float:
    """Compute threshold-based anomaly score.

    If optional_threshold is provided, check if max(incident) exceeds it.
    Otherwise, use baseline 0.99 quantile as dynamic threshold.
    Returns a score in [0, 1].
    """
    cfg = config or AnomalyConfig()
    if len(incident) == 0:
        return 0.0
    if optional_threshold is not None:
        threshold = optional_threshold
    else:
        threshold = float(np.quantile(baseline, 0.99))
    max_val = float(np.max(incident))
    if threshold <= cfg.eps:
        return 0.0
    exceed_ratio = max(0.0, (max_val - threshold) / (abs(threshold) + cfg.eps))
    return min(1.0, exceed_ratio)


def change_rate_score(
    incident: np.ndarray,
    baseline: np.ndarray,
    config: Optional[AnomalyConfig] = None,
) -> float:
    """Compute change-rate anomaly score.

    Calculates adjacent-point change rates and compares incident max rate
    against baseline rate distribution. Returns score in [0, 1].
    """
    cfg = config or AnomalyConfig()
    if len(incident) < 2 or len(baseline) < 2:
        return 0.0

    def _change_rates(x: np.ndarray) -> np.ndarray:
        return np.abs(np.diff(x)) / (np.abs(x[:-1]) + cfg.eps)

    incident_rates = _change_rates(incident)
    baseline_rates = _change_rates(baseline)
    max_incident_rate = float(np.max(incident_rates))
    if len(baseline_rates) == 0:
        return min(1.0, max_incident_rate)
    bl_median, bl_mad = robust_median_mad(baseline_rates)
    bl_scale = 1.4826 * bl_mad + cfg.eps
    z = (max_incident_rate - bl_median) / bl_scale
    return min(1.0, max(0.0, z / 5.0))


def max_deviation_score(
    incident: np.ndarray,
    baseline: np.ndarray,
    config: Optional[AnomalyConfig] = None,
) -> float:
    """Compute maximum deviation score of incident from baseline median.

    Returns normalized score in [0, 1].
    """
    cfg = config or AnomalyConfig()
    if len(incident) == 0 or len(baseline) == 0:
        return 0.0
    median, mad = robust_median_mad(baseline)
    scale = 1.4826 * mad + cfg.eps
    max_dev = float(np.max(np.abs(incident - median)))
    return min(1.0, max_dev / scale / 5.0)


def traditional_anomaly_score(
    incident: np.ndarray,
    baseline: np.ndarray,
    optional_threshold: Optional[float] = None,
    config: Optional[AnomalyConfig] = None,
) -> Tuple[float, float, float, float, float]:
    """Fuse traditional anomaly detection methods into a single score.

    Returns:
        (final_score, threshold_sc, zscore_sc, change_rate_sc, deviation_sc)
    """
    cfg = config or AnomalyConfig()
    thr_sc = threshold_score(incident, baseline, optional_threshold, cfg)
    z_scores = robust_z_scores(incident, baseline, cfg)
    zscore_sc = min(1.0, float(np.max(np.abs(z_scores))) / 5.0)
    cr_sc = change_rate_score(incident, baseline, cfg)
    dev_sc = max_deviation_score(incident, baseline, cfg)

    final = (
        cfg.traditional_weight_threshold * thr_sc
        + cfg.traditional_weight_zscore * zscore_sc
        + cfg.traditional_weight_change_rate * cr_sc
        + cfg.traditional_weight_max_deviation * dev_sc
    )
    return final, thr_sc, zscore_sc, cr_sc, dev_sc
