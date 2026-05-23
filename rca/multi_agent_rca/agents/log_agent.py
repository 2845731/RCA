from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional

import pandas as pd

from rca.multi_agent_rca.core.dataset import telemetry_dir
from rca.multi_agent_rca.core.io_utils import normalize_time_column
from rca.multi_agent_rca.core.schema import EvidenceReport, RCAQuery
from rca.multi_agent_rca.core.time_utils import day_dir_from_epoch


class LogAgent:
    """Semantic log evidence agent with lightweight Drain-like templating."""

    DYNAMIC_RE = re.compile(r"0x[0-9a-fA-F]+|[a-fA-F0-9]{16,}|\d+\.\d+|\d+")

    def run(self, query: RCAQuery, selected_templates: Optional[Iterable[str]] = None) -> EvidenceReport:
        log_dir = telemetry_dir(query.dataset) / day_dir_from_epoch(query.start_ts) / "log"
        if not log_dir.exists():
            return EvidenceReport(
                agent_name="LogAgent",
                evidence_type="log",
                confidence=0.0,
                support=["No log directory for this dataset/date; using metric and trace evidence only."],
            )

        selected = set(selected_templates or [])
        component_scores: Dict[str, float] = defaultdict(float)
        reason_scores: Dict[str, float] = defaultdict(float)
        template_counts: Counter = Counter()
        support: List[str] = []
        total_rows = 0
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
                    work = chunk[(chunk[time_col] >= query.start_ts) & (chunk[time_col] <= query.end_ts)].copy()
                    if work.empty:
                        continue
                    total_rows += len(work)
                    comp_col = "cmdb_id" if "cmdb_id" in work.columns else None
                    for _, row in work.iterrows():
                        component = str(row[comp_col]) if comp_col else "system"
                        template = self.template(str(row.get("value", "")))
                        template_id = self.template_id(template)
                        if selected and template_id not in selected:
                            continue
                        template_counts[template_id] += 1
                        reason, weight = self._reason_from_template(query.candidate_reasons, template)
                        if reason:
                            component_scores[component] += weight
                            reason_scores[reason] += weight
            except Exception as exc:
                support.append(f"Skipped {path.name}: {exc}")

        candidates = [
            {"component": comp, "score": round(score, 4)}
            for comp, score in sorted(component_scores.items(), key=lambda x: x[1], reverse=True)[:20]
        ]
        support.append(f"Analyzed {total_rows} log rows and {len(template_counts)} templates.")
        confidence = min(1.0, sum(component_scores.values()) / 20.0) if component_scores else 0.05
        return EvidenceReport(
            agent_name="LogAgent",
            evidence_type="log",
            candidates=candidates,
            confidence=confidence,
            support=support,
            raw_refs=[str(log_dir)],
            details={
                "component_scores": dict(component_scores),
                "reason_scores": dict(reason_scores),
                "template_counts": dict(template_counts.most_common(50)),
            },
        )

    def template(self, value: str) -> str:
        compact = " ".join(value.lower().split())
        return self.DYNAMIC_RE.sub("<*>", compact)

    def template_id(self, template: str) -> str:
        return hashlib.sha1(template.encode("utf-8", errors="ignore")).hexdigest()[:12]

    def _reason_from_template(self, candidate_reasons: Iterable[str], template: str) -> tuple[Optional[str], float]:
        candidates = list(candidate_reasons)
        rules = [
            (["outofmemory", "oom", "heap"], ["JVM Out of Memory (OOM) Heap", "high memory usage"], 3.0),
            (["gc", "cms", "memory"], ["high memory usage", "JVM Out of Memory (OOM) Heap"], 1.0),
            (["cpu"], ["high CPU usage", "CPU fault", "high JVM CPU load"], 1.5),
            (["timeout", "latency", "delay", "slow"], ["network latency", "network delay"], 2.0),
            (["packet", "loss", "reset"], ["network packet loss", "network loss"], 2.0),
            (["connection", "connect", "too many"], ["db connection limit", "db close"], 2.0),
            (["disk", "i/o", "io ", "read", "write"], ["high disk I/O read usage", "high disk space usage"], 1.5),
            (["error", "exception", "fail", "warn"], candidates, 0.5),
        ]
        for keys, reasons, weight in rules:
            if any(key in template for key in keys):
                for reason in reasons:
                    if reason in candidates:
                        return reason, weight
        return None, 0.0
