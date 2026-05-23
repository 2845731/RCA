from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from rca.multi_agent_rca.core.dataset import telemetry_dir
from rca.multi_agent_rca.core.io_utils import normalize_time_column, read_csv_safe
from rca.multi_agent_rca.core.schema import AnomalySegment, EvidenceReport, RCAQuery
from rca.multi_agent_rca.core.time_utils import day_dir_from_epoch, epoch_to_local


class MetricAgent:
    """Metric evidence agent.

    It calculates global thresholds from the full metric file first, then slices
    the query window. This follows the rule in the original RCA-agent prompt and
    avoids local-window threshold leakage.
    """

    NUMERIC_EXCLUDE = {"timestamp", "startTime", "itemid", "bomc_id", "index"}
    KPI_COLUMNS = ("kpi_name", "name")
    COMPONENT_COLUMNS = ("cmdb_id", "tc", "serviceName", "service")

    def __init__(self, max_segments: int = 80) -> None:
        self.max_segments = max_segments

    def run(self, query: RCAQuery) -> EvidenceReport:
        metric_dir = telemetry_dir(query.dataset) / day_dir_from_epoch(query.start_ts) / "metric"
        if not metric_dir.exists():
            return EvidenceReport(
                agent_name="MetricAgent",
                evidence_type="metric",
                confidence=0.0,
                contradiction=[f"Metric directory not found: {metric_dir}"],
            )

        segments: List[AnomalySegment] = []
        support: List[str] = []
        for path in sorted(metric_dir.glob("*.csv")):
            try:
                df = read_csv_safe(path)
                time_col = normalize_time_column(df)
                if time_col is None:
                    continue
                for component, kpi, series in self._iter_series(df, path.name, time_col):
                    segs = self._detect_segments(component, kpi, path.name, series, query)
                    segments.extend(segs)
            except Exception as exc:
                support.append(f"Skipped {path.name}: {exc}")

        segments = sorted(segments, key=lambda s: s.severity, reverse=True)[: self.max_segments]
        candidates = [s.to_dict() for s in segments[:30]]
        component_scores: Dict[str, float] = defaultdict(float)
        reason_scores: Dict[str, float] = defaultdict(float)
        for seg in segments:
            component_scores[seg.component] += seg.severity
            if seg.reason_hint:
                reason_scores[seg.reason_hint] += seg.severity

        if segments:
            support.append(
                f"Detected {len(segments)} metric anomaly segments in {query.start_time} to {query.end_time}."
            )
        else:
            support.append("No strong metric anomaly segment found; downstream agents should rely more on trace/log.")

        confidence = min(1.0, sum(s.severity for s in segments[:5]) / 25.0) if segments else 0.05
        return EvidenceReport(
            agent_name="MetricAgent",
            evidence_type="metric",
            candidates=candidates,
            confidence=confidence,
            support=support,
            raw_refs=[str(metric_dir)],
            details={
                "segments": candidates,
                "component_scores": dict(component_scores),
                "reason_scores": dict(reason_scores),
            },
        )

    def _iter_series(
        self, df: pd.DataFrame, file_name: str, time_col: str
    ) -> Iterable[Tuple[str, str, pd.DataFrame]]:
        if "value" in df.columns and any(c in df.columns for c in self.KPI_COLUMNS):
            kpi_col = next(c for c in self.KPI_COLUMNS if c in df.columns)
            comp_col = "cmdb_id" if "cmdb_id" in df.columns else None
            work = df[[time_col, kpi_col, "value"] + ([comp_col] if comp_col else [])].copy()
            work["value"] = pd.to_numeric(work["value"], errors="coerce")
            work = work.dropna(subset=["value", time_col, kpi_col])
            group_cols = ([comp_col] if comp_col else []) + [kpi_col]
            for keys, group in work.groupby(group_cols):
                if not isinstance(keys, tuple):
                    keys = (keys,)
                component = str(keys[0]) if comp_col else self._component_from_file(file_name)
                kpi = str(keys[-1])
                yield component, kpi, group[[time_col, "value"]].rename(columns={time_col: "timestamp"})
            return

        comp_col = next((c for c in self.COMPONENT_COLUMNS if c in df.columns), None)
        numeric_cols = []
        for col in df.columns:
            if col in self.NUMERIC_EXCLUDE or col == comp_col:
                continue
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() > 0:
                numeric_cols.append(col)
        if not numeric_cols:
            return
        if comp_col:
            for component, group in df.groupby(comp_col):
                for col in numeric_cols:
                    series = pd.DataFrame(
                        {
                            "timestamp": group[time_col],
                            "value": pd.to_numeric(group[col], errors="coerce"),
                        }
                    ).dropna()
                    yield str(component), str(col), series
        else:
            for col in numeric_cols:
                series = pd.DataFrame(
                    {"timestamp": df[time_col], "value": pd.to_numeric(df[col], errors="coerce")}
                ).dropna()
                yield self._component_from_file(file_name), str(col), series

    def _component_from_file(self, file_name: str) -> str:
        return Path(file_name).stem.replace("metric_", "")

    def _detect_segments(
        self, component: str, kpi: str, file_name: str, series: pd.DataFrame, query: RCAQuery
    ) -> List[AnomalySegment]:
        series = series.dropna().sort_values("timestamp")
        if len(series) < 4:
            return []
        values = pd.to_numeric(series["value"], errors="coerce").dropna()
        if values.empty:
            return []
        q05 = float(values.quantile(0.05))
        q10 = float(values.quantile(0.10))
        q50 = float(values.quantile(0.50))
        q90 = float(values.quantile(0.90))
        q95 = float(values.quantile(0.95))
        iqr = float(values.quantile(0.75) - values.quantile(0.25))
        std = float(values.std()) if len(values) > 1 else 0.0
        scale = max(iqr / 1.349 if iqr else 0.0, std, abs(q95 - q05) / 3.0, abs(q50) * 0.05, 1e-6)

        window = series[(series["timestamp"] >= query.start_ts) & (series["timestamp"] <= query.end_ts)].copy()
        if window.empty:
            return []

        reason_hint = self._reason_hint(query.candidate_reasons, kpi, file_name)
        low_sensitive = self._is_low_sensitive(kpi)
        rows = []
        for _, row in window.iterrows():
            value = float(row["value"])
            high_score = max(0.0, (value - q95) / scale)
            low_threshold = q10 if low_sensitive else q05
            low_score = max(0.0, (low_threshold - value) / scale)
            absolute_score = self._absolute_domain_score(kpi, value, q50, q95)
            if absolute_score:
                high_score = max(high_score, absolute_score)
            direction = "high" if high_score >= low_score else "low"
            score = max(high_score, low_score)
            score = self._cap_spiky_counter_score(kpi, value, q95, score, reason_hint)
            if score >= 0.35:
                rows.append(
                    {
                        "timestamp": int(row["timestamp"]),
                        "value": value,
                        "score": float(score),
                        "direction": direction,
                        "threshold": q95 if direction == "high" else low_threshold,
                    }
                )
        if not rows:
            return []

        rows = sorted(rows, key=lambda r: r["timestamp"])
        gaps = series["timestamp"].diff().dropna()
        median_gap = int(gaps.median()) if not gaps.empty else 60
        max_gap = max(180, median_gap * 3)
        groups: List[List[Dict[str, float]]] = []
        current = [rows[0]]
        for row in rows[1:]:
            if row["timestamp"] - current[-1]["timestamp"] <= max_gap:
                current.append(row)
            else:
                groups.append(current)
                current = [row]
        groups.append(current)

        segments = []
        for group in groups:
            severity = sum(float(r["score"]) for r in group) + max(float(r["score"]) for r in group)
            severity *= self._persistence_weight(kpi, group, median_gap, reason_hint)
            peak = max(group, key=lambda r: abs(float(r["score"])))
            first = group[0]
            segments.append(
                AnomalySegment(
                    component=component,
                    kpi=kpi,
                    file_name=file_name,
                    start_ts=int(first["timestamp"]),
                    end_ts=int(group[-1]["timestamp"]),
                    start_time=epoch_to_local(int(first["timestamp"])),
                    end_time=epoch_to_local(int(group[-1]["timestamp"])),
                    direction=str(peak["direction"]),
                    severity=round(float(severity), 4),
                    max_value=float(peak["value"]),
                    threshold=float(peak["threshold"]),
                    reason_hint=reason_hint,
                    evidence_rows=group[:8],
                )
            )
        return segments

    def _is_low_sensitive(self, kpi: str) -> bool:
        low = kpi.lower()
        return any(key in low for key in ["success", "succee", "sr", "rate", "rr", "avail", "idle"])

    def _absolute_domain_score(self, kpi: str, value: float, q50: float, q95: float) -> float:
        """Score sustained saturation even when the whole-day quantile is flat.

        Some OpenRCA injections keep resource metrics pinned near saturation, so
        pure quantile scoring can miss them. These weak absolute priors are kept
        below severe z-score spikes and mainly help persistent memory/disk/cpu
        pressure surface as candidate evidence.
        """
        low = kpi.lower()
        is_percent = any(key in low for key in ["percent", "perc", "pct", "util"])
        steady_high_baseline = q50 >= 85 and q95 >= 88
        if any(key in low for key in ["free", "idle", "avail"]):
            if is_percent and value <= 10:
                return min(3.0, 0.75 + (10.0 - value) / 4.0)
            return 0.0
        if "mem" in low or "memory" in low or "heap" in low:
            if is_percent and value >= 88 and not steady_high_baseline:
                return min(4.0, 0.75 + (value - 88.0) / 4.0)
        if "cpu" in low and is_percent and value >= 85 and not steady_high_baseline:
            return min(3.5, 0.5 + (value - 85.0) / 5.0)
        if any(key in low for key in ["dskpercentbusy", "diskpercentbusy"]) and value >= 80:
            return min(3.0, 0.5 + (value - 80.0) / 8.0)
        return 0.0

    def _cap_spiky_counter_score(
        self,
        kpi: str,
        value: float,
        q95: float,
        score: float,
        reason_hint: Optional[str],
    ) -> float:
        low = kpi.lower()
        if q95 <= 1e-9 and abs(value) < 1.0:
            cap = 2.5 if reason_hint else 1.75
            score = min(score, cap)
        if "zabbix_agentd" in low:
            score = min(score, 1.5)
        return float(score)

    def _persistence_weight(
        self,
        kpi: str,
        group: List[Dict[str, float]],
        median_gap: int,
        reason_hint: Optional[str],
    ) -> float:
        low = kpi.lower()
        points = len(group)
        duration = int(group[-1]["timestamp"]) - int(group[0]["timestamp"])
        if points <= 1:
            weight = 0.28
        elif points == 2:
            weight = 0.65
        elif duration >= max(240, median_gap * 4):
            weight = 1.25
        else:
            weight = 1.0
        if reason_hint is None:
            weight *= 0.85
        if "zabbix_agentd" in low:
            weight *= 0.35
        return weight

    def _reason_hint(self, candidate_reasons: Iterable[str], kpi: str, file_name: str) -> Optional[str]:
        low = f"{kpi} {file_name}".lower()
        candidates = list(candidate_reasons)
        rules = [
            (["oom", "heap"], ["JVM Out of Memory (OOM) Heap", "high memory usage"]),
            (["jvm", "cpuload"], ["high JVM CPU load", "high CPU usage", "CPU fault"]),
            (["cpu", "proc_user"], ["high CPU usage", "CPU fault"]),
            (["mem", "memory"], ["high memory usage"]),
            (["packet", "loss", "fin-wait", "close-wait"], ["network packet loss", "network loss", "db close"]),
            (["latency", "delay", "mrt", "time", "avg_time", "tnsping"], ["network latency", "network delay"]),
            (["connect", "sess"], ["db connection limit"]),
            (["disk", "i/o", "io_", "iowait", "read"], ["high disk I/O read usage", "high disk space usage"]),
            (["space", "used"], ["high disk space usage", "high memory usage"]),
        ]
        for keys, reasons in rules:
            if any(key in low for key in keys):
                for reason in reasons:
                    if reason in candidates:
                        return reason
        return None
