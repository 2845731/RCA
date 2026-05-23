from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def resample_series(
    df: pd.DataFrame,
    time_col: str = "timestamp",
    value_col: str = "value",
    interval: str = "60s",
) -> pd.DataFrame:
    """Resample a time series to a uniform interval using mean aggregation.

    Args:
        df: DataFrame with time_col and value_col.
        time_col: Name of the timestamp column.
        value_col: Name of the value column.
        interval: Resampling interval string (e.g. '60s', '5min').

    Returns:
        Resampled DataFrame with timestamp and value columns.
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], unit="s", errors="coerce")
    df = df.dropna(subset=[time_col])
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.set_index(time_col)
    df = df.resample(interval).mean()
    df = df.reset_index()
    df = df.rename(columns={time_col: "timestamp", value_col: "value"})
    return df[["timestamp", "value"]]


def fill_missing(
    series: np.ndarray,
    method: str = "linear",
) -> np.ndarray:
    """Fill missing values (NaN) in a numpy array.

    Args:
        series: Input array potentially containing NaN values.
        method: Interpolation method ('linear', 'ffill', 'zero').

    Returns:
        Array with NaN values filled.
    """
    if not np.any(np.isnan(series)):
        return series

    s = pd.Series(series)
    if method == "linear":
        s = s.interpolate(method="linear")
    elif method == "ffill":
        s = s.ffill()
    elif method == "zero":
        s = s.fillna(0.0)
    else:
        s = s.interpolate(method="linear")

    s = s.ffill().bfill()
    return s.values


def extract_windows(
    df: pd.DataFrame,
    incident_start: str,
    incident_end: str,
    baseline_multiplier: int = 3,
) -> Tuple[np.ndarray, np.ndarray, str, str]:
    """Extract incident and baseline windows from a time series DataFrame.

    The baseline window is the period immediately before the incident,
    with length = baseline_multiplier * incident_length.

    Args:
        df: DataFrame with 'timestamp' and 'value' columns.
        incident_start: Start time string for incident window.
        incident_end: End time string for incident window.
        baseline_multiplier: Multiplier for baseline window length.

    Returns:
        (incident_values, baseline_values, baseline_start, baseline_end)
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    inc_start = pd.to_datetime(incident_start)
    inc_end = pd.to_datetime(incident_end)
    incident_len = inc_end - inc_start
    baseline_end_dt = inc_start
    baseline_start_dt = inc_start - baseline_multiplier * incident_len

    incident_mask = (df["timestamp"] >= inc_start) & (df["timestamp"] <= inc_end)
    baseline_mask = (df["timestamp"] >= baseline_start_dt) & (df["timestamp"] < baseline_end_dt)

    incident_values = df.loc[incident_mask, "value"].dropna().values
    baseline_values = df.loc[baseline_mask, "value"].dropna().values

    return (
        incident_values,
        baseline_values,
        str(baseline_start_dt),
        str(baseline_end_dt),
    )


def normalize_series(series: np.ndarray, method: str = "minmax") -> Tuple[np.ndarray, Dict[str, float]]:
    """Normalize a series and return normalization parameters.

    Args:
        series: Input array.
        method: Normalization method ('minmax' or 'zscore').

    Returns:
        (normalized_series, params_dict)
    """
    if method == "minmax":
        vmin = float(np.min(series))
        vmax = float(np.max(series))
        rng = vmax - vmin
        if rng < 1e-12:
            return np.zeros_like(series), {"min": vmin, "max": vmax}
        normalized = (series - vmin) / rng
        return normalized, {"min": vmin, "max": vmax}
    elif method == "zscore":
        mean = float(np.mean(series))
        std = float(np.std(series))
        if std < 1e-12:
            return np.zeros_like(series), {"mean": mean, "std": std}
        normalized = (series - mean) / std
        return normalized, {"mean": mean, "std": std}
    else:
        return series.copy(), {}
