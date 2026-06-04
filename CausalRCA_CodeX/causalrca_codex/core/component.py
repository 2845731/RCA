from __future__ import annotations

import re
from typing import Iterable, Optional


def infer_component_type(component: str) -> str:
    value = str(component)
    low = value.lower()
    if low.startswith("os_") or low.startswith("node-") or low in {"host", "node"}:
        return "node"
    if "mysql" in low or low.startswith("db_") or low.startswith("db"):
        return "db"
    if "redis" in low:
        return "redis"
    if low.startswith("docker_") or re.search(r"-\d+$", low) or "." in low:
        return "pod"
    if any(key in low for key in ["tomcat", "apache", "service", "frontend", "cart", "checkout"]):
        return "service"
    if low.startswith("mg") or low.startswith("ig") or "middleware" in low or "gateway" in low:
        return "middleware"
    return "service"


def normalize_component_id(raw: object, candidates: Iterable[str]) -> str:
    value = str(raw).strip()
    candidate_set = set(str(c).strip() for c in candidates)
    if value in candidate_set:
        return value

    if "." in value:
        left, right = value.split(".", 1)
        if left in candidate_set:
            return left
        right_head = right.split(".", 1)[0]
        if right_head in candidate_set:
            return right_head

    service = value
    for suffix in ("-grpc", ".ts:8088", ".ts:8080", ".source", ".destination"):
        if suffix in service:
            service = service.split(suffix, 1)[0]
            if service in candidate_set:
                return service

    if ":" in value:
        head = value.split(":", 1)[0].split(".", 1)[0]
        if head in candidate_set:
            return head

    match = re.search(r"([A-Za-z][A-Za-z0-9_-]+-\d+)", value)
    if match and match.group(1) in candidate_set:
        return match.group(1)

    for candidate in candidate_set:
        if candidate and candidate in value:
            return candidate
    return value


def nearest_candidate(raw: object, candidates: Iterable[str]) -> Optional[str]:
    normalized = normalize_component_id(raw, candidates)
    return normalized if normalized in set(candidates) else None
