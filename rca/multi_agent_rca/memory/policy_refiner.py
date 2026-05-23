from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

from rca.multi_agent_rca.core.io_utils import read_jsonl


class PolicyRefiner:
    """Offline prompt/rule distillation helper.

    The class does not mutate prompts automatically. It produces a versioned
    rule proposal that can be reviewed before being used in a new experiment.
    """

    def propose_rules(self, case_memory_path: Path) -> Dict[str, List[str]]:
        rows = read_jsonl(case_memory_path)
        reason_counter = Counter()
        for row in rows:
            if row.get("score") == 1.0:
                reason_counter.update(row.get("reasons", []))
        positive = [f"Prioritize evidence patterns that previously supported reason: {r}" for r, _ in reason_counter.most_common(10)]
        negative = [
            "Audit contradictions between loud metric symptoms and trace downstream evidence.",
            "When logs are absent, lower confidence for reason-only tasks.",
        ]
        return {"positive_rules": positive, "negative_rules": negative}
