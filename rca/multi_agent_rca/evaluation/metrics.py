from __future__ import annotations

import re
from datetime import datetime
from typing import Dict, List


def element_accuracy(prediction: str, scoring_points: str) -> Dict[str, float]:
    """Compute field-level accuracy for OpenRCA scoring strings."""

    parsed = _parse_prediction(prediction)
    components = re.findall(r"The (?:\d+-th|only) predicted root cause component is ([^\n]+)", scoring_points)
    reasons = re.findall(r"The (?:\d+-th|only) predicted root cause reason is ([^\n]+)", scoring_points)
    times = re.findall(
        r"The (?:\d+-th|only) root cause occurrence time is within 1 minutes \(i.e., <=1min\) of ([^\n]+)",
        scoring_points,
    )
    out = {}
    if components:
        out["component_accuracy"] = _list_accuracy([p.get("root cause component", "") for p in parsed], components)
    if reasons:
        out["reason_accuracy"] = _list_accuracy([p.get("root cause reason", "") for p in parsed], reasons)
    if times:
        preds = [p.get("root cause occurrence datetime", "") for p in parsed]
        out["time_accuracy"] = _time_accuracy(preds, times)
    return out


def _parse_prediction(prediction: str) -> List[Dict[str, str]]:
    pattern = (
        r'{\s*'
        r'(?:"root cause occurrence datetime":\s*"(.*?)")?,?\s*'
        r'(?:"root cause component":\s*"(.*?)")?,?\s*'
        r'(?:"root cause reason":\s*"(.*?)")?\s*}'
    )
    rows = []
    for dt, comp, reason in re.findall(pattern, prediction):
        rows.append(
            {
                "root cause occurrence datetime": dt,
                "root cause component": comp,
                "root cause reason": reason,
            }
        )
    return rows


def _list_accuracy(preds: List[str], labels: List[str]) -> float:
    if not labels:
        return 0.0
    return sum(1 for label in labels if label in preds) / len(labels)


def _time_accuracy(preds: List[str], labels: List[str]) -> float:
    if not labels:
        return 0.0
    correct = 0
    for label in labels:
        for pred in preds:
            try:
                if abs(
                    datetime.strptime(pred, "%Y-%m-%d %H:%M:%S")
                    - datetime.strptime(label, "%Y-%m-%d %H:%M:%S")
                ).total_seconds() <= 60:
                    correct += 1
                    break
            except Exception:
                continue
    return correct / len(labels)
