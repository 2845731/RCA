from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping

from causalrca_codex.core.component import infer_component_type


def _text(*parts: object) -> str:
    return " ".join(str(part or "") for part in parts).lower()


def _has(text: str, tokens: Iterable[str]) -> bool:
    return any(token in text for token in tokens)


def signal_family(name: str, file_name: str = "") -> str:
    """Map KPI/log/reason text to a portable evidence family.

    This is intentionally semantic and dataset-agnostic: families describe
    fault physics (packet loss, disk capacity, heap pressure), not specific
    component names or benchmark labels.
    """
    low = _text(name, file_name)
    if _has(low, ["oom", "outofmemory"]):
        return "jvm_oom"
    if _has(low, ["jvm", "heap", "gc", "fullgc"]) and _has(low, ["cpu", "load"]):
        return "jvm_cpu"
    if _has(low, ["heap", "jvm"]) and _has(low, ["used", "free", "max", "total", "mem", "memory"]):
        return "jvm_memory"
    if _has(low, ["packet", "loss", "retrans", "drop", "corrupt"]):
        return "network_loss"
    if _has(low, ["latency", "delay", "timeout", "elapsed", "mrt", "response", "avg_time", "fin-wait", "close-wait"]):
        return "network_latency"
    if _has(low, ["disk", "filesystem", "space", "capacity"]) and _has(low, ["space", "used", "free", "capacity", "filesystem"]):
        return "disk_space"
    if _has(low, ["disk", "i/o", "io_", "iowait", "read", "write", "throughput", "busy"]):
        return "disk_io"
    if _has(low, ["mysql", "jdbc", "sql", "innodb", "qcache", "db", "database"]):
        if _has(low, ["connect", "client", "pool", "session", "limit"]):
            return "db_connection"
        if _has(low, ["close", "closed"]):
            return "db_close"
        return "db"
    if _has(low, ["cpu", "processor", "load", "singlecpu", "cpuload"]):
        return "cpu"
    if _has(low, ["mem", "memory", "rss", "swap", "cachemem", "usermem"]):
        return "memory"
    if _has(low, ["restart", "terminated", "killed", "kill", "crash", "exit"]):
        return "process"
    return "other"


def reason_family(reason: str) -> str:
    low = _text(reason)
    if "oom" in low or ("heap" in low and "memory" in low):
        return "jvm_oom"
    if "jvm" in low and "cpu" in low:
        return "jvm_cpu"
    if "packet" in low or "loss" in low or "corrupt" in low:
        return "network_loss"
    if "latency" in low or "delay" in low or "timeout" in low:
        return "network_latency"
    if "disk" in low and ("space" in low or "capacity" in low):
        return "disk_space"
    if "disk" in low or "i/o" in low or "io" in low:
        return "disk_io"
    if "connection" in low or "limit" in low:
        return "db_connection"
    if "db close" in low or re.search(r"\bclose\b", low):
        return "db_close"
    if "memory" in low or "mem" in low:
        return "memory"
    if "cpu" in low or "load" in low:
        return "cpu"
    if "process" in low or "termination" in low:
        return "process"
    return "other"


def family_alignment_score(reason: str, family: str) -> float:
    rf = reason_family(reason)
    if rf == family:
        return 1.0

    broad = {
        "jvm_oom": "memory",
        "jvm_memory": "memory",
        "memory": "memory",
        "jvm_cpu": "cpu",
        "cpu": "cpu",
        "network_loss": "network",
        "network_latency": "network",
        "disk_space": "disk",
        "disk_io": "disk",
        "db_connection": "db",
        "db_close": "db",
        "db": "db",
    }
    if broad.get(rf) and broad.get(rf) == broad.get(family):
        return 0.45
    if rf == "jvm_oom" and family == "jvm_memory":
        return 0.75
    if rf == "jvm_cpu" and family == "cpu":
        return 0.65
    return 0.0


def semantic_match_score(reason: str, kpi: str, file_name: str = "") -> float:
    return family_alignment_score(reason, signal_family(kpi, file_name))


def signal_ownership_weight(component: str, kpi: str, file_name: str = "") -> float:
    """Estimate whether a signal belongs to the component's own fault mechanism.

    OS-level resource metrics are useful context, but in service systems they are
    often shared propagation symptoms. Component-native metrics such as Redis,
    Tomcat/JVM, and MySQL counters carry stronger local causal evidence. This
    keeps the rule mechanism-level rather than benchmark-label-specific.
    """
    low = _text(kpi, file_name)
    comp_type = infer_component_type(component)

    if comp_type == "db" and _has(low, ["mysql", "sql", "innodb", "qcache", "database"]):
        return 1.0
    if comp_type == "redis" and "redis" in low:
        return 1.0
    if comp_type in {"service", "middleware"} and _has(low, ["tomcat", "apache", "jvm", "heap", "gc"]):
        return 1.0
    if comp_type in {"service", "middleware"} and _has(
        low,
        [
            "latency",
            "delay",
            "timeout",
            "packet",
            "loss",
            "network",
            "fin-wait",
            "close-wait",
            "time-wait",
            "mrt",
            "response",
            "elapsed",
        ],
    ):
        return 0.95
    if comp_type in {"pod", "node"} and _has(low, ["container", "docker", "k8s", "process", "oslinux"]):
        return 0.85
    if _has(low, ["oslinux", "localdisk", "singlecpu", "cpuidle", "cpuwio", "network", "memory_"]):
        return 0.45
    return 0.65


def build_component_profiles(
    anomaly_details: Mapping[str, List[Dict[str, Any]]],
    candidate_reasons: Iterable[str],
) -> Dict[str, Dict[str, Any]]:
    """Create component-level causal evidence profiles.

    The profile uses inverse component frequency over evidence families. A
    family that appears on every component is treated as propagation context;
    a family concentrated on one/few components is stronger local root-cause
    evidence. This avoids hard-coding any benchmark-specific component.
    """
    reasons = list(candidate_reasons)
    raw_family: Dict[str, Dict[str, float]] = {}
    raw_reason: Dict[str, Dict[str, float]] = {}

    for component, segments in anomaly_details.items():
        family_scores: Dict[str, float] = defaultdict(float)
        reason_scores: Dict[str, float] = defaultdict(float)
        for segment in segments:
            severity = max(0.0, float(segment.get("severity", segment.get("max_deviation", 0.0)) or 0.0))
            if severity <= 0:
                continue
            kpi = str(segment.get("kpi", ""))
            file_name = str(segment.get("file_name", ""))
            ownership = signal_ownership_weight(component, kpi, file_name)
            owned_severity = severity * ownership
            family = signal_family(kpi, file_name)
            family_scores[family] += owned_severity
            hinted = segment.get("reason_hint")
            if hinted in reasons:
                reason_scores[str(hinted)] += owned_severity
            for reason in reasons:
                match = semantic_match_score(reason, kpi, file_name)
                if match > 0:
                    reason_scores[reason] += owned_severity * match
        raw_family[component] = dict(family_scores)
        raw_reason[component] = dict(reason_scores)

    component_count = max(len(raw_family), 1)
    family_df: Dict[str, int] = defaultdict(int)
    for family_scores in raw_family.values():
        for family, score in family_scores.items():
            if score > 0:
                family_df[family] += 1
    family_idf = {
        family: 1.0 + math.log((1.0 + component_count) / (1.0 + df))
        for family, df in family_df.items()
    }

    profiles: Dict[str, Dict[str, Any]] = {}
    max_reason_raw = max(
        (score for reason_map in raw_reason.values() for score in reason_map.values()),
        default=1.0,
    )
    max_local_raw = 0.0
    local_raw_by_component: Dict[str, float] = {}

    for component, family_scores in raw_family.items():
        weighted_families = {
            family: score * family_idf.get(family, 1.0)
            for family, score in family_scores.items()
        }
        local_raw = max(weighted_families.values(), default=0.0)
        local_raw_by_component[component] = local_raw
        max_local_raw = max(max_local_raw, local_raw)

    for component, family_scores in raw_family.items():
        reason_map = raw_reason.get(component, {})
        reason_evidence = {
            reason: min(1.0, reason_map.get(reason, 0.0) / max(max_reason_raw, 1e-9))
            for reason in reasons
        }
        profiles[component] = {
            "reason_evidence": reason_evidence,
            "local_root_evidence": min(1.0, local_raw_by_component.get(component, 0.0) / max(max_local_raw, 1e-9)),
        }
    return profiles
