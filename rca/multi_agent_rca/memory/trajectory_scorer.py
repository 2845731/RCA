from __future__ import annotations

from typing import Any, Dict, Iterable, List


def score_trajectory(trajectory: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    """Score diagnostic process quality independently from answer correctness."""

    states = [str(item.get("state", "")) for item in trajectory]
    state_set = set(states)
    coverage = sum(1 for s in ["DISPATCH", "SYNTHESIZE", "REFLECT", "FINALIZE"] if s in state_set) / 4.0
    recheck = 1.0 if "RECHECK" in state_set else 0.0
    debate = 1.0 if "DEBATE" in state_set else 0.0
    return {
        "state_coverage": coverage,
        "used_recheck": recheck,
        "used_debate": debate,
        "steps": float(len(states)),
    }
