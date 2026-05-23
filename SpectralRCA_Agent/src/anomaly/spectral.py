from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.anomaly.traditional import robust_median_mad
from src.config import AnomalyConfig


def center_signal(x: np.ndarray) -> np.ndarray:
    """Center signal by subtracting its mean."""
    return x - np.mean(x)


def compute_fft_power(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute FFT power spectrum of a 1D signal.

    Args:
        x: 1D input signal (incident series).

    Returns:
        (X, power_no_dc, freqs) where X is the raw FFT output,
        power_no_dc excludes the DC component, and freqs are the frequencies.
    """
    x_centered = center_signal(x)
    n = len(x_centered)
    X = np.fft.rfft(x_centered)
    freqs = np.fft.rfftfreq(n)
    power = np.abs(X) ** 2
    power_no_dc = power[1:]
    freqs_no_dc = freqs[1:]
    return X, power_no_dc, freqs_no_dc


@dataclass
class SpectralFeatures:
    total_energy: float
    low_ratio: float
    mid_ratio: float
    high_ratio: float
    dominant_freq_index: int
    dominant_freq_energy_ratio: float
    spectral_entropy: float


def compute_spectral_features(x: np.ndarray) -> SpectralFeatures:
    """Compute spectral features from a 1D signal.

    Low/mid/high frequency bands are defined by splitting power_no_dc into thirds.
    """
    _, power_no_dc, _ = compute_fft_power(x)
    if len(power_no_dc) == 0:
        return SpectralFeatures(
            total_energy=0.0, low_ratio=0.0, mid_ratio=0.0, high_ratio=0.0,
            dominant_freq_index=0, dominant_freq_energy_ratio=0.0, spectral_entropy=0.0,
        )

    total_energy = float(np.sum(power_no_dc))
    if total_energy < 1e-12:
        return SpectralFeatures(
            total_energy=0.0, low_ratio=0.0, mid_ratio=0.0, high_ratio=0.0,
            dominant_freq_index=0, dominant_freq_energy_ratio=0.0, spectral_entropy=0.0,
        )

    n = len(power_no_dc)
    third = max(1, n // 3)
    low_end = third
    mid_end = min(2 * third, n)

    low_energy = float(np.sum(power_no_dc[:low_end]))
    mid_energy = float(np.sum(power_no_dc[low_end:mid_end]))
    high_energy = float(np.sum(power_no_dc[mid_end:]))

    low_ratio = low_energy / total_energy
    mid_ratio = mid_energy / total_energy
    high_ratio = high_energy / total_energy

    dominant_idx = int(np.argmax(power_no_dc))
    dominant_freq_energy_ratio = float(power_no_dc[dominant_idx]) / total_energy

    normalized = power_no_dc / total_energy
    normalized = normalized[normalized > 0]
    spectral_entropy = float(-np.sum(normalized * np.log2(normalized))) if len(normalized) > 0 else 0.0

    return SpectralFeatures(
        total_energy=total_energy,
        low_ratio=low_ratio,
        mid_ratio=mid_ratio,
        high_ratio=high_ratio,
        dominant_freq_index=dominant_idx,
        dominant_freq_energy_ratio=dominant_freq_energy_ratio,
        spectral_entropy=spectral_entropy,
    )


def split_baseline_windows(baseline: np.ndarray, window_len: int) -> List[np.ndarray]:
    """Split baseline series into windows of length window_len.

    If baseline is shorter than window_len, use the entire baseline as one window.
    """
    if len(baseline) < window_len:
        return [baseline.copy()]
    windows = []
    for start in range(0, len(baseline) - window_len + 1, window_len):
        windows.append(baseline[start:start + window_len].copy())
    if len(windows) == 0:
        windows.append(baseline[:window_len].copy())
    return windows


@dataclass
class BaselineSpectralStats:
    total_energy_median: float
    total_energy_mad: float
    low_ratio_median: float
    low_ratio_mad: float
    mid_ratio_median: float
    mid_ratio_mad: float
    high_ratio_median: float
    high_ratio_mad: float
    dominant_freq_energy_ratio_median: float
    dominant_freq_energy_ratio_mad: float
    spectral_entropy_median: float
    spectral_entropy_mad: float


def compute_baseline_spectral_stats(baseline_windows: List[np.ndarray]) -> BaselineSpectralStats:
    """Compute median and MAD of spectral features across baseline windows."""
    if not baseline_windows:
        return BaselineSpectralStats(
            total_energy_median=0.0, total_energy_mad=0.0,
            low_ratio_median=0.0, low_ratio_mad=0.0,
            mid_ratio_median=0.0, mid_ratio_mad=0.0,
            high_ratio_median=0.0, high_ratio_mad=0.0,
            dominant_freq_energy_ratio_median=0.0, dominant_freq_energy_ratio_mad=0.0,
            spectral_entropy_median=0.0, spectral_entropy_mad=0.0,
        )

    features_list = [compute_spectral_features(w) for w in baseline_windows]
    n = len(features_list)

    def _med_mad(vals: List[float]) -> Tuple[float, float]:
        arr = np.array(vals)
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        return med, mad

    te_med, te_mad = _med_mad([f.total_energy for f in features_list])
    lr_med, lr_mad = _med_mad([f.low_ratio for f in features_list])
    mr_med, mr_mad = _med_mad([f.mid_ratio for f in features_list])
    hr_med, hr_mad = _med_mad([f.high_ratio for f in features_list])
    df_med, df_mad = _med_mad([f.dominant_freq_energy_ratio for f in features_list])
    se_med, se_mad = _med_mad([f.spectral_entropy for f in features_list])

    return BaselineSpectralStats(
        total_energy_median=te_med, total_energy_mad=te_mad,
        low_ratio_median=lr_med, low_ratio_mad=lr_mad,
        mid_ratio_median=mr_med, mid_ratio_mad=mr_mad,
        high_ratio_median=hr_med, high_ratio_mad=hr_mad,
        dominant_freq_energy_ratio_median=df_med, dominant_freq_energy_ratio_mad=df_mad,
        spectral_entropy_median=se_med, spectral_entropy_mad=se_mad,
    )


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-x))
    else:
        ex = np.exp(x)
        return ex / (1.0 + ex)


def spectral_anomaly_score(
    incident: np.ndarray,
    baseline: np.ndarray,
    config: Optional[AnomalyConfig] = None,
) -> Tuple[float, SpectralFeatures, Dict[str, float], BaselineSpectralStats]:
    """Compute spectral anomaly score for incident vs baseline.

    Returns:
        (spectral_score, incident_features, z_values_dict, baseline_stats)
    """
    cfg = config or AnomalyConfig()
    incident_features = compute_spectral_features(incident)
    window_len = len(incident)
    baseline_windows = split_baseline_windows(baseline, window_len)
    baseline_stats = compute_baseline_spectral_stats(baseline_windows)

    eps = cfg.eps

    def _robust_z(value: float, median: float, mad: float) -> float:
        scale = 1.4826 * mad + eps
        return (value - median) / scale

    spectral_energy_z = _robust_z(
        np.log(incident_features.total_energy + eps),
        np.log(baseline_stats.total_energy_median + eps),
        baseline_stats.total_energy_mad,
    )
    low_ratio_z = _robust_z(incident_features.low_ratio, baseline_stats.low_ratio_median, baseline_stats.low_ratio_mad)
    high_ratio_z = _robust_z(incident_features.high_ratio, baseline_stats.high_ratio_median, baseline_stats.high_ratio_mad)
    dominant_freq_z = _robust_z(
        incident_features.dominant_freq_energy_ratio,
        baseline_stats.dominant_freq_energy_ratio_median,
        baseline_stats.dominant_freq_energy_ratio_mad,
    )
    entropy_z = _robust_z(incident_features.spectral_entropy, baseline_stats.spectral_entropy_median, baseline_stats.spectral_entropy_mad)

    energy_score = _sigmoid((spectral_energy_z - cfg.spectral_energy_z_threshold) / 1.0)
    low_shift_score = _sigmoid((abs(low_ratio_z) - cfg.spectral_ratio_z_threshold) / 1.0)
    high_shift_score = _sigmoid((abs(high_ratio_z) - cfg.spectral_ratio_z_threshold) / 1.0)
    periodic_score = _sigmoid((dominant_freq_z - cfg.spectral_ratio_z_threshold) / 1.0)
    entropy_shift_score = _sigmoid((abs(entropy_z) - cfg.spectral_ratio_z_threshold) / 1.0)

    spectral_score = (
        cfg.spectral_weight_energy * energy_score
        + cfg.spectral_weight_low_shift * low_shift_score
        + cfg.spectral_weight_high_shift * high_shift_score
        + cfg.spectral_weight_periodic * periodic_score
        + cfg.spectral_weight_entropy * entropy_shift_score
    )

    z_values = {
        "spectral_energy_z": spectral_energy_z,
        "low_ratio_z": low_ratio_z,
        "high_ratio_z": high_ratio_z,
        "dominant_freq_energy_ratio_z": dominant_freq_z,
        "entropy_z": entropy_z,
    }

    return spectral_score, incident_features, z_values, baseline_stats


def classify_spectral_anomaly(
    features: SpectralFeatures,
    z_values: Dict[str, float],
    config: Optional[AnomalyConfig] = None,
) -> str:
    """Classify the type of spectral anomaly based on features and z-values."""
    cfg = config or AnomalyConfig()
    spectral_energy_z = z_values.get("spectral_energy_z", 0.0)

    if spectral_energy_z < cfg.spectral_energy_z_threshold:
        return "no_strong_spectral_anomaly"

    if features.low_ratio >= 0.55:
        return "slow_trend_anomaly"
    if features.high_ratio >= 0.45:
        return "fast_burst_or_jitter_anomaly"
    if features.dominant_freq_energy_ratio >= 0.60:
        return "periodic_oscillation_anomaly"

    return "mixed_spectral_anomaly"
