from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Set

import pandas as pd

from rca.multi_agent_rca.core.dataset import telemetry_dir
from rca.multi_agent_rca.core.io_utils import normalize_time_column
from rca.multi_agent_rca.core.schema import EvidenceReport, RCAQuery
from rca.multi_agent_rca.core.time_utils import day_dir_from_epoch


class TraceAgent:
    """Trace propagation evidence agent."""

    def run(self, query: RCAQuery, metric_report: EvidenceReport, selected_trace_ids: Optional[Iterable[str]] = None) -> EvidenceReport:
        trace_dir = telemetry_dir(query.dataset) / day_dir_from_epoch(query.start_ts) / "trace"
        if not trace_dir.exists():
            return EvidenceReport(
                agent_name="TraceAgent",
                evidence_type="trace",
                confidence=0.0,
                contradiction=[f"Trace directory not found: {trace_dir}"],
            )

        selected = set(str(x) for x in selected_trace_ids or [])
        metric_components = set(metric_report.details.get("component_scores", {}).keys())
        edge_counts: Counter = Counter()
        self_calls: Counter = Counter()
        slow_components: Counter = Counter()
        failed_components: Counter = Counter()
        trace_rows = 0
        support: List[str] = []

        for path in sorted(trace_dir.glob("*.csv")):
            try:
                for chunk in pd.read_csv(path, chunksize=200_000, low_memory=False):
                    time_col = normalize_time_column(chunk)
                    if time_col is None:
                        continue
                    if chunk[time_col].min() > query.end_ts:
                        break
                    if chunk[time_col].max() < query.start_ts:
                        continue
                    trace_col = self._first_col(chunk, ["traceId", "trace_id"])
                    span_col = self._first_col(chunk, ["id", "span_id"])
                    parent_col = self._first_col(chunk, ["pid", "parent_id", "parent_span"])
                    duration_col = self._first_col(chunk, ["elapsedTime", "duration"])
                    comp_col = "cmdb_id" if "cmdb_id" in chunk.columns else None
                    if not all([trace_col, span_col, parent_col, comp_col]):
                        support.append(f"Skipped {path.name}: missing trace/span/parent/component columns")
                        break

                    work = chunk[(chunk[time_col] >= query.start_ts) & (chunk[time_col] <= query.end_ts)].copy()
                    if selected and trace_col:
                        work = work[work[trace_col].astype(str).isin(selected)]
                    if work.empty:
                        continue
                    trace_rows += len(work)
                    id_to_comp = dict(zip(work[span_col].astype(str), work[comp_col].astype(str)))
                    q95 = 0.0
                    if duration_col:
                        duration_series = pd.to_numeric(work[duration_col], errors="coerce")
                        q95 = float(duration_series.quantile(0.95)) if duration_series.notna().any() else 0.0

                    for _, row in work.iterrows():
                        comp = str(row[comp_col])
                        parent_id = str(row[parent_col])
                        parent_comp = id_to_comp.get(parent_id)
                        if parent_comp:
                            edge_counts[(parent_comp, comp)] += 1
                            if parent_comp == comp:
                                self_calls[comp] += 1
                        if duration_col and q95:
                            duration = pd.to_numeric(pd.Series([row[duration_col]]), errors="coerce").iloc[0]
                            if pd.notna(duration) and float(duration) > q95:
                                slow_components[comp] += 1
                        if "success" in work.columns and str(row.get("success", "")).lower() in {"false", "0", "fail", "failed"}:
                            failed_components[comp] += 1
            except Exception as exc:
                support.append(f"Skipped {path.name}: {exc}")

        component_scores: Dict[str, float] = defaultdict(float)
        for (src, dst), count in edge_counts.items():
            if dst in metric_components and src in metric_components:
                component_scores[dst] += min(5.0, 0.08 * count)
            elif dst in metric_components:
                component_scores[dst] += min(2.0, 0.03 * count)
        for comp, count in self_calls.items():
            component_scores[comp] += min(4.0, 0.15 * count)
        for comp, count in slow_components.items():
            component_scores[comp] += min(4.0, 0.08 * count)
        for comp, count in failed_components.items():
            component_scores[comp] += min(4.0, 0.2 * count)

        top_edges = [
            {"source": src, "target": dst, "count": count}
            for (src, dst), count in edge_counts.most_common(30)
        ]
        candidates = [
            {"component": comp, "score": round(score, 4)}
            for comp, score in sorted(component_scores.items(), key=lambda x: x[1], reverse=True)[:20]
        ]
        support.append(f"Analyzed {trace_rows} trace rows and {len(edge_counts)} component edges.")
        confidence = min(1.0, (sum(component_scores.values()) / 15.0)) if component_scores else 0.05
        return EvidenceReport(
            agent_name="TraceAgent",
            evidence_type="trace",
            candidates=candidates,
            confidence=confidence,
            support=support,
            raw_refs=[str(trace_dir)],
            details={
                "component_scores": dict(component_scores),
                "top_edges": top_edges,
                "self_calls": dict(self_calls),
                "slow_components": dict(slow_components),
                "failed_components": dict(failed_components),
            },
        )

    def _first_col(self, df: pd.DataFrame, options: List[str]) -> Optional[str]:
        for col in options:
            if col in df.columns:
                return col
        return None
