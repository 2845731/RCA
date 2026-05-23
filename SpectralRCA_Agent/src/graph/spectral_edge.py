from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from src.anomaly.spectral import compute_spectral_features, compute_fft_power, center_signal
from src.config import GraphConfig


def compute_cross_correlation(
    source: np.ndarray,
    target: np.ndarray,
    max_lag: Optional[int] = None,
) -> Tuple[float, int]:
    """Compute normalized cross-correlation between source and target series.

    Returns:
        (best_correlation, best_lag) where lag > 0 means target lags source.
    """
    n = min(len(source), len(target))
    if n < 3:
        return 0.0, 0

    s = source[:n] - np.mean(source[:n])
    t = target[:n] - np.mean(target[:n])

    s_std = np.std(s)
    t_std = np.std(t)
    if s_std < 1e-12 or t_std < 1e-12:
        return 0.0, 0

    if max_lag is None:
        max_lag = max(1, n // 4)

    best_corr = 0.0
    best_lag = 0

    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            s_shifted = s[:n - lag] if lag > 0 else s
            t_shifted = t[lag:] if lag > 0 else t
        else:
            s_shifted = s[-lag:]
            t_shifted = t[:n + lag]

        min_len = min(len(s_shifted), len(t_shifted))
        if min_len < 3:
            continue

        corr = np.sum(s_shifted[:min_len] * t_shifted[:min_len]) / (
            min_len * s_std * t_std + 1e-12
        )
        if abs(corr) > abs(best_corr):
            best_corr = corr
            best_lag = lag

    return float(best_corr), best_lag


def compute_spectral_shape_similarity(
    source: np.ndarray,
    target: np.ndarray,
) -> float:
    """Compute spectral shape similarity between two series.

    Uses cosine similarity of normalized power spectra.
    Returns value in [0, 1] where 1 means identical spectral shapes.
    """
    _, power_s, _ = compute_fft_power(source)
    _, power_t, _ = compute_fft_power(target)

    if len(power_s) == 0 or len(power_t) == 0:
        return 0.0

    min_len = min(len(power_s), len(power_t))
    ps = power_s[:min_len]
    pt = power_t[:min_len]

    ps_norm = ps / (np.linalg.norm(ps) + 1e-12)
    pt_norm = pt / (np.linalg.norm(pt) + 1e-12)

    cosine_sim = float(np.dot(ps_norm, pt_norm))
    return max(0.0, min(1.0, (cosine_sim + 1.0) / 2.0))


def compute_dominant_freq_match(
    source: np.ndarray,
    target: np.ndarray,
    tolerance: int = 1,
) -> float:
    """Check if source and target have matching dominant frequencies.

    Args:
        source, target: Input series.
        tolerance: Number of frequency bins tolerance for matching.

    Returns:
        1.0 if dominant frequencies match within tolerance, 0.0 otherwise.
    """
    feat_s = compute_spectral_features(source)
    feat_t = compute_spectral_features(target)

    if abs(feat_s.dominant_freq_index - feat_t.dominant_freq_index) <= tolerance:
        return 1.0
    return 0.0


def compute_phase_lag_consistency(
    source: np.ndarray,
    target: np.ndarray,
    best_lag: int,
    min_length: int = 16,
) -> float:
    """Compute phase lag consistency between source and target.

    Validates that the time-domain cross-correlation lag is consistent
    with the phase difference at the dominant frequency.

    Returns:
        Consistency score in [0, 1].
    """
    n = min(len(source), len(target))
    if n < min_length:
        return 0.5

    X_s = np.fft.rfft(center_signal(source[:n]))
    X_t = np.fft.rfft(center_signal(target[:n]))

    freqs = np.fft.rfftfreq(n)
    power = np.abs(X_s) ** 2 + np.abs(X_t) ** 2

    if len(power) <= 1:
        return 0.5

    dominant_idx = int(np.argmax(power[1:])) + 1
    phase_s = np.angle(X_s[dominant_idx])
    phase_t = np.angle(X_t[dominant_idx])

    phase_diff = phase_t - phase_s
    phase_diff = (phase_diff + np.pi) % (2 * np.pi) - np.pi

    if freqs[dominant_idx] < 1e-12:
        return 0.5

    expected_lag_samples = phase_diff / (2 * np.pi * freqs[dominant_idx])

    if expected_lag_samples == 0:
        return 1.0 if best_lag == 0 else 0.0

    lag_ratio = best_lag / expected_lag_samples if abs(expected_lag_samples) > 1e-6 else 0.0

    if lag_ratio > 0:
        consistency = min(1.0, 1.0 / (1.0 + abs(lag_ratio - 1.0)))
    else:
        consistency = 0.0

    return consistency


def compute_spectral_edge_score(
    source: np.ndarray,
    target: np.ndarray,
    best_lag: int,
    config: Optional[GraphConfig] = None,
) -> Tuple[float, float, float, float]:
    """Compute the spectral edge score combining all spectral validation metrics.

    Returns:
        (spectral_edge_score, shape_similarity, freq_match, phase_consistency)
    """
    cfg = config or GraphConfig()

    shape_sim = compute_spectral_shape_similarity(source, target)
    freq_match = compute_dominant_freq_match(source, target, cfg.freq_tolerance)
    phase_cons = compute_phase_lag_consistency(source, target, best_lag, cfg.min_phase_lag_length)

    spectral_score = (
        cfg.edge_spectral_weight_shape * shape_sim
        + cfg.edge_spectral_weight_freq * freq_match
        + cfg.edge_spectral_weight_phase * phase_cons
    )

    return spectral_score, shape_sim, freq_match, phase_cons
