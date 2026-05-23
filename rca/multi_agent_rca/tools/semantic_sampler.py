from __future__ import annotations

import hashlib
import math
import time
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

from rca.multi_agent_rca.agents.log_agent import LogAgent
from rca.multi_agent_rca.core.dataset import telemetry_dir
from rca.multi_agent_rca.core.io_utils import normalize_time_column
from rca.multi_agent_rca.core.schema import EvidenceReport, RCAQuery, SamplerReport
from rca.multi_agent_rca.core.time_utils import day_dir_from_epoch


class SemanticSampler:
    """Gleaner-inspired sampler for OpenRCA telemetry windows.

    It keeps diagnostically rich traces by balancing anomaly focus and EPS
    diversity. The selected trace IDs are passed into TraceAgent, while log
    template IDs can constrain LogAgent. Metrics are not sampled here because
    MetricAgent needs full-day data to calculate global thresholds correctly.
    """

    def __init__(self, max_traces: int = 160, max_log_templates: int = 80) -> None:
        self.max_traces = max_traces
        self.max_log_templates = max_log_templates
        self.log_agent = LogAgent()

    def run(self, query: RCAQuery, metric_report: EvidenceReport) -> SamplerReport:
        t0 = time.time()
        metric_scores = metric_report.details.get("component_scores", {})
        trace_ids, pattern_counts, trace_rows = self._sample_traces(query, metric_scores)
        templates, log_rows = self._sample_logs(query)
        total_patterns = len(pattern_counts)
        selected_patterns = len({p for _, p, _ in trace_ids})
        selected_trace_ids = [tid for tid, _, _ in trace_ids]
        sampled_rows = len(selected_trace_ids) + len(templates)
        entropy = self._entropy([pattern for _, pattern, _ in trace_ids])
        anomaly_ratio = 0.0
        if trace_ids:
            anomaly_ratio = sum(1 for _, _, score in trace_ids if score > 0) / len(trace_ids)
        return SamplerReport(
            selected_trace_ids=selected_trace_ids,
            selected_log_templates=templates,
            trace_pattern_coverage=selected_patterns / total_patterns if total_patterns else 0.0,
            shannon_entropy=entropy,
            anomaly_ratio=anomaly_ratio,
            input_rows=trace_rows + log_rows,
            sampled_rows=sampled_rows,
            elapsed_seconds=round(time.time() - t0, 4),
            notes=[
                "Metric files are not sampled because thresholding needs full-day series.",
                "Trace EPS diversity uses greedy Jaccard approximation.",
            ],
        )

    def _sample_traces(self, query: RCAQuery, metric_scores: Dict[str, float]) -> Tuple[List[Tuple[str, str, float]], Counter, int]:
        trace_dir = telemetry_dir(query.dataset) / day_dir_from_epoch(query.start_ts) / "trace"
        if not trace_dir.exists():
            return [], Counter(), 0
        candidates: List[Tuple[str, str, Set[str], float, str]] = []
        input_rows = 0
        pattern_counts: Counter = Counter()
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
                    comp_col = "cmdb_id" if "cmdb_id" in chunk.columns else None
                    if not all([trace_col, span_col, parent_col, comp_col]):
                        break
                    work = chunk[(chunk[time_col] >= query.start_ts) & (chunk[time_col] <= query.end_ts)].copy()
                    input_rows += len(work)
                    if work.empty:
                        continue
                    for trace_id, group in work.groupby(trace_col):
                        eps = self._trace_eps(group, span_col, parent_col, comp_col)
                        if not eps:
                            continue
                        pattern = self._pattern_id(eps)
                        pattern_counts[pattern] += 1
                        anomaly = self._trace_anomaly_score(group, comp_col, metric_scores)
                        root = self._root_component(group, span_col, parent_col, comp_col)
                        candidates.append((str(trace_id), pattern, eps, anomaly, root))
            except Exception:
                continue
        if not candidates:
            return [], pattern_counts, input_rows

        by_root: Dict[str, List[Tuple[str, str, Set[str], float, str]]] = defaultdict(list)
        for item in candidates:
            by_root[item[4]].append(item)

        per_group_budget = max(1, self.max_traces // max(1, len(by_root)))
        selected: List[Tuple[str, str, float]] = []
        for _, group in by_root.items():
            selected.extend(self._greedy_select(group, per_group_budget))
        if len(selected) < self.max_traces:
            remaining = [c for c in candidates if c[0] not in {s[0] for s in selected}]
            selected.extend(self._greedy_select(remaining, self.max_traces - len(selected)))
        return selected[: self.max_traces], pattern_counts, input_rows

    def _sample_logs(self, query: RCAQuery) -> Tuple[List[str], int]:
        log_dir = telemetry_dir(query.dataset) / day_dir_from_epoch(query.start_ts) / "log"
        if not log_dir.exists():
            return [], 0
        scores: Counter = Counter()
        rows = 0
        for path in sorted(log_dir.glob("*.csv")):
            try:
                for chunk in pd.read_csv(path, chunksize=120_000, low_memory=False):
                    time_col = normalize_time_column(chunk)
                    if time_col is None or "value" not in chunk.columns:
                        continue
                    if chunk[time_col].min() > query.end_ts:
                        break
                    if chunk[time_col].max() < query.start_ts:
                        continue
                    work = chunk[(chunk[time_col] >= query.start_ts) & (chunk[time_col] <= query.end_ts)]
                    rows += len(work)
                    for value in work["value"].astype(str):
                        template = self.log_agent.template(value)
                        template_id = self.log_agent.template_id(template)
                        weight = 1
                        low = template.lower()
                        if any(k in low for k in ["error", "exception", "fail", "oom", "timeout", "warn"]):
                            weight += 4
                        if any(k in low for k in ["gc", "memory", "connect", "packet", "disk"]):
                            weight += 2
                        scores[template_id] += weight
            except Exception:
                continue
        return [tpl for tpl, _ in scores.most_common(self.max_log_templates)], rows

    def _trace_eps(self, group: pd.DataFrame, span_col: str, parent_col: str, comp_col: str) -> Set[str]:
        id_to_comp = dict(zip(group[span_col].astype(str), group[comp_col].astype(str)))
        eps = set()
        for _, row in group.iterrows():
            comp = str(row[comp_col])
            parent_comp = id_to_comp.get(str(row[parent_col]))
            if parent_comp:
                eps.add(f"call:{parent_comp}->{comp}")
            if "success" in group.columns and str(row.get("success", "")).lower() in {"false", "0", "fail", "failed"}:
                eps.add(f"status_error:{comp}")
        return eps

    def _trace_anomaly_score(self, group: pd.DataFrame, comp_col: str, metric_scores: Dict[str, float]) -> float:
        score = 0.0
        for comp in group[comp_col].astype(str).unique():
            score += float(metric_scores.get(comp, 0.0))
        if "success" in group.columns:
            score += 2.0 * sum(str(v).lower() in {"false", "0", "fail", "failed"} for v in group["success"])
        return score

    def _root_component(self, group: pd.DataFrame, span_col: str, parent_col: str, comp_col: str) -> str:
        span_ids = set(group[span_col].astype(str))
        roots = group[~group[parent_col].astype(str).isin(span_ids)]
        if not roots.empty:
            return str(roots.iloc[0][comp_col])
        return str(group.iloc[0][comp_col])

    def _greedy_select(self, candidates: List[Tuple[str, str, Set[str], float, str]], budget: int) -> List[Tuple[str, str, float]]:
        selected: List[Tuple[str, str, Set[str], float, str]] = []
        remaining = sorted(candidates, key=lambda x: x[3], reverse=True)
        while remaining and len(selected) < budget:
            best_idx = 0
            best_score = -1.0
            for idx, item in enumerate(remaining[:300]):
                diversity = 1.0
                if selected:
                    diversity = 1.0 - max(self._jaccard(item[2], s[2]) for s in selected)
                score = item[3] + diversity
                if score > best_score:
                    best_score = score
                    best_idx = idx
            selected.append(remaining.pop(best_idx))
        return [(tid, pattern, anomaly) for tid, pattern, _, anomaly, _ in selected]

    def _jaccard(self, a: Set[str], b: Set[str]) -> float:
        if not a and not b:
            return 1.0
        return len(a & b) / max(1, len(a | b))

    def _pattern_id(self, eps: Set[str]) -> str:
        joined = "\n".join(sorted(eps))
        return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:12]

    def _entropy(self, items: List[str]) -> float:
        if not items:
            return 0.0
        counts = Counter(items)
        total = sum(counts.values())
        return -sum((c / total) * math.log(c / total + 1e-12, 2) for c in counts.values())

    def _first_col(self, df: pd.DataFrame, options: List[str]) -> Optional[str]:
        for col in options:
            if col in df.columns:
                return col
        return None
