from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from rca.multi_agent_rca.core.dataset import project_root
from rca.multi_agent_rca.core.io_utils import append_jsonl, read_jsonl
from rca.multi_agent_rca.core.schema import RCAQuery, RCAResult


class MemoryStore:
    """Case-level memory for self-evolving RCA experiments."""

    def __init__(self, memory_dir: Optional[Path] = None, frozen: bool = True) -> None:
        self.memory_dir = memory_dir or (project_root() / "test" / "multi_agent_rca" / "memory")
        self.frozen = frozen
        self.case_path = self.memory_dir / "case_memory.jsonl"
        self.bad_path = self.memory_dir / "bad_trajectory_bank.jsonl"
        self.rule_path = self.memory_dir / "failure_pattern_rules.json"
        self.rules = self._load_rules()

    def retrieve(self, query: RCAQuery, metric_report: Dict[str, Any], top_k: int = 3) -> List[Dict[str, Any]]:
        cases = read_jsonl(self.case_path)
        if not cases:
            return []
        query_tokens = self._tokens(query.instruction)
        metric_components = set(metric_report.get("component_scores", {}).keys())
        scored = []
        for case in cases:
            score = 0.0
            score += len(query_tokens & set(case.get("tokens", []))) / max(len(query_tokens), 1)
            score += 0.2 if case.get("dataset") == query.dataset else 0.0
            score += 0.1 * len(metric_components & set(case.get("components", [])))
            scored.append((score, case))
        return [case for score, case in sorted(scored, key=lambda x: x[0], reverse=True)[:top_k] if score > 0]

    def record(self, query: RCAQuery, result: RCAResult, score: Optional[float] = None) -> None:
        if self.frozen:
            return
        components = [c.get("component") for c in result.ranked_candidates if c.get("component")]
        reasons = [c.get("reason") for c in result.ranked_candidates if c.get("reason")]
        row = {
            "dataset": query.dataset,
            "row_id": query.row_id,
            "task_index": query.task_index,
            "tokens": sorted(self._tokens(query.instruction)),
            "components": components,
            "reasons": reasons,
            "score": score,
            "prediction": result.prediction_json,
        }
        append_jsonl(self.case_path, row)
        if score is not None and score < 1.0:
            append_jsonl(self.bad_path, {"query": row, "trajectory": result.trajectory})

    def _load_rules(self) -> Dict[str, Any]:
        if not self.rule_path.exists():
            return {
                "negative_rules": [
                    "Do not compute anomaly thresholds after filtering to the query window.",
                    "Do not select a downstream component unless it is also faulty in metric/log evidence.",
                    "Prefer the first point of a persistent anomaly segment as occurrence time.",
                ]
            }
        with self.rule_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _tokens(self, text: str) -> set[str]:
        return {tok for tok in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split() if len(tok) > 2}
