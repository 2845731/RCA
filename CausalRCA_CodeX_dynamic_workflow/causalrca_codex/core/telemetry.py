from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from causalrca_codex.config import AgentLoopConfig
from causalrca_codex.core.component import infer_component_type, normalize_component_id
from causalrca_codex.core.dataset import telemetry_path
from causalrca_codex.core.time_utils import day_dir_from_epoch, normalize_epoch
from causalrca_codex.schemas import MetricSeries, RCAQuery, TelemetryFrame


TIMESTAMP_COLUMNS = ("timestamp", "startTime", "time", "datetime")
COMPONENT_COLUMNS = ("cmdb_id", "tc", "serviceName", "service")
KPI_COLUMNS = ("kpi_name", "name")
NUMERIC_EXCLUDE = {
    "timestamp",
    "startTime",
    "time",
    "datetime",
    "itemid",
    "bomc_id",
    "index",
    "traceId",
    "trace_id",
    "span_id",
    "parent_id",
    "pid",
    "id",
}


def read_csv_safe(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def normalize_time_column(df: pd.DataFrame) -> Optional[str]:
    for col in TIMESTAMP_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(normalize_epoch)
            df.dropna(subset=[col], inplace=True)
            df[col] = df[col].astype(int)
            return col
    return None


def load_day_frames(config: AgentLoopConfig, query: RCAQuery) -> Dict[str, List[TelemetryFrame]]:
    day_dir = telemetry_path(config, query.dataset) / day_dir_from_epoch(query.start_ts)
    frames: Dict[str, List[TelemetryFrame]] = {"metric": [], "trace": [], "log": []}
    for kind in frames:
        folder = day_dir / kind
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.csv")):
            df = read_csv_safe(path)
            time_col = normalize_time_column(df)
            frames[kind].append(
                TelemetryFrame(
                    path=str(path),
                    kind=kind,
                    file_name=path.name,
                    rows=len(df),
                    timestamp_col=time_col,
                    data=df,
                )
            )
    return frames


def iter_metric_series(
    frames: Iterable[TelemetryFrame],
    query: RCAQuery,
    aggregate: str = "mean",
    threshold_percentile: float = 95.0,
    low_percentile: float = 5.0,
    window_start_ts: Optional[int] = None,
) -> List[MetricSeries]:
    series_list: List[MetricSeries] = []
    effective_start = window_start_ts if window_start_ts is not None else query.start_ts
    for frame in frames:
        if frame.timestamp_col is None:
            continue
        df = frame.data
        for raw_component, component, kpi, full in _iter_component_kpis(df, frame.file_name, frame.timestamp_col, query):
            if full.empty:
                continue
            full = full.dropna().sort_values("timestamp")
            full["value"] = pd.to_numeric(full["value"], errors="coerce")
            full = full.dropna(subset=["value"])
            if full.empty:
                continue
            if aggregate == "max":
                full = full.groupby("timestamp", as_index=False)["value"].max()
            else:
                full = full.groupby("timestamp", as_index=False)["value"].mean()
            window = full[(full["timestamp"] >= effective_start) & (full["timestamp"] <= query.end_ts)].copy()
            values = full["value"].astype(float)
            high = float(values.quantile(threshold_percentile / 100.0))
            low = float(values.quantile(low_percentile / 100.0))
            median = float(values.quantile(0.50))
            iqr = float(values.quantile(0.75) - values.quantile(0.25))
            std = float(values.std()) if len(values) > 1 else 0.0
            scale = max(iqr / 1.349 if iqr else 0.0, std, abs(high - low) / 3.0, abs(median) * 0.05, 1e-6)

            # --- Filter constant/dead metrics ---
            # Constant metrics (e.g., HeapMemoryMax, JVMMaxMemory) never change.
            # Their scale collapses to near-zero, causing deviation = (value - threshold)/scale
            # to explode to huge numbers. Detect and skip these metrics.
            cv = std / max(abs(median), 1e-6)  # coefficient of variation
            if cv < 0.001 and iqr < 1e-4:
                # This is a constant/dead metric - skip it entirely
                continue
            series_list.append(
                MetricSeries(
                    component=component,
                    raw_component=raw_component,
                    component_type=infer_component_type(component),
                    kpi=kpi,
                    file_name=frame.file_name,
                    full=full,
                    window=window,
                    threshold_high=high,
                    threshold_low=low,
                    median=median,
                    scale=scale,
                )
            )
    return series_list


def _iter_component_kpis(
    df: pd.DataFrame,
    file_name: str,
    time_col: str,
    query: RCAQuery,
) -> Iterable[Tuple[str, str, str, pd.DataFrame]]:
    if "value" in df.columns and any(col in df.columns for col in KPI_COLUMNS):
        kpi_col = next(col for col in KPI_COLUMNS if col in df.columns)
        comp_col = next((col for col in COMPONENT_COLUMNS if col in df.columns), None)
        use_cols = [time_col, kpi_col, "value"] + ([comp_col] if comp_col else [])
        work = df[use_cols].copy()
        work["value"] = pd.to_numeric(work["value"], errors="coerce")
        work = work.dropna(subset=[time_col, kpi_col, "value"])
        group_cols = ([comp_col] if comp_col else []) + [kpi_col]
        for keys, group in work.groupby(group_cols):
            if not isinstance(keys, tuple):
                keys = (keys,)
            raw_component = str(keys[0]) if comp_col else Path(file_name).stem
            component = normalize_component_id(raw_component, query.candidate_components)
            kpi = str(keys[-1])
            yield raw_component, component, kpi, group[[time_col, "value"]].rename(columns={time_col: "timestamp"})
        return

    comp_col = next((col for col in COMPONENT_COLUMNS if col in df.columns), None)
    numeric_cols = []
    for col in df.columns:
        if col == comp_col or col in NUMERIC_EXCLUDE:
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() > 0:
            numeric_cols.append(col)

    if comp_col:
        for raw_component, group in df.groupby(comp_col):
            component = normalize_component_id(raw_component, query.candidate_components)
            for col in numeric_cols:
                values = pd.to_numeric(group[col], errors="coerce")
                full = pd.DataFrame({"timestamp": group[time_col], "value": values}).dropna()
                yield str(raw_component), component, str(col), full
    else:
        raw_component = Path(file_name).stem.replace("metric_", "")
        component = normalize_component_id(raw_component, query.candidate_components)
        for col in numeric_cols:
            values = pd.to_numeric(df[col], errors="coerce")
            full = pd.DataFrame({"timestamp": df[time_col], "value": values}).dropna()
            yield raw_component, component, str(col), full


def window_frame(frame: TelemetryFrame, start_ts: int, end_ts: int) -> pd.DataFrame:
    if frame.timestamp_col is None:
        return frame.data
    return frame.data[
        (frame.data[frame.timestamp_col] >= start_ts) & (frame.data[frame.timestamp_col] <= end_ts)
    ].copy()
