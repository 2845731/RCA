from __future__ import annotations

from typing import Iterable, Optional


def is_low_sensitive_kpi(kpi: str) -> bool:
    low = kpi.lower()
    return any(key in low for key in ["success", "succee", "sr", "rr", "rate", "avail", "idle"])


def reason_hint(candidate_reasons: Iterable[str], kpi: str, file_name: str = "") -> Optional[str]:
    low = f"{kpi} {file_name}".lower()
    candidates = list(candidate_reasons)
    rules = [
        (["oom", "heap"], ["JVM Out of Memory (OOM) Heap", "high memory usage", "container memory load"]),
        (["jvm", "cpuload"], ["high JVM CPU load", "high CPU usage", "CPU fault", "container CPU load"]),
        (["cpu", "proc_user", "cpu_used", "cpucpu", "cpuutil"], ["high CPU usage", "CPU fault", "container CPU load", "node CPU load"]),
        (["mem", "memory"], ["high memory usage", "container memory load", "node memory consumption"]),
        (["packet", "loss"], ["network packet loss", "network loss", "container packet loss"]),
        (["fin-wait", "close-wait", "time-wait"], ["network packet loss", "network latency", "network delay"]),
        (["latency", "delay", "mrt", "time", "avg_time", "elapsed", "slow", "timeout"], ["network latency", "network delay", "container network latency"]),
        (["connect", "sess", "client"], ["db connection limit"]),
        (["close"], ["db close"]),
        (["disk", "i/o", "io_", "iowait", "read"], ["high disk I/O read usage", "node disk read I/O consumption"]),
        (["write"], ["node disk write I/O consumption", "container write I/O load"]),
        (["space", "used"], ["high disk space usage", "node disk space consumption"]),
        (["restart", "terminated", "kill"], ["container process termination"]),
        (["corrupt"], ["container network packet corruption"]),
        (["iowait"], ["high disk I/O read usage", "CPU fault"]),
    ]
    for keys, reasons in rules:
        if any(key in low for key in keys):
            for reason in reasons:
                if reason in candidates:
                    return reason
    return None


def absolute_domain_deviation(kpi: str, value: float, median: float, high_threshold: float) -> float:
    low = kpi.lower()
    is_percent = any(key in low for key in ["percent", "pct", "perc", "util", "rate"])
    steady_high_baseline = median >= 85 and high_threshold >= 88
    if any(key in low for key in ["free", "idle", "avail"]):
        if is_percent and value <= 10:
            return min(1.0, 0.25 + (10.0 - value) / 12.0)
        return 0.0
    if any(key in low for key in ["mem", "memory", "heap"]):
        if is_percent and value >= 95:
            return min(1.0, 0.50 + (value - 95.0) / 10.0)
        if is_percent and value >= 88 and not steady_high_baseline:
            return min(1.0, 0.25 + (value - 88.0) / 12.0)
        if is_percent and value >= 90 and steady_high_baseline:
            return min(0.8, 0.20 + (value - 90.0) / 15.0)
    if "cpu" in low and is_percent and value >= 90:
        return min(1.0, 0.35 + (value - 90.0) / 10.0)
    if "cpu" in low and is_percent and value >= 85 and not steady_high_baseline:
        return min(1.0, 0.20 + (value - 85.0) / 15.0)
    if any(key in low for key in ["dskpercentbusy", "diskpercentbusy"]) and value >= 80:
        return min(1.0, 0.20 + (value - 80.0) / 20.0)
    return 0.0


def reason_prior(reason: str, component_type: str) -> float:
    low = reason.lower()
    if component_type in {"db", "redis"} and any(key in low for key in ["db", "connection", "close", "limit"]):
        return 0.85
    if component_type in {"db", "redis"} and any(key in low for key in ["memory", "oom", "heap", "mem"]):
        return 0.80
    if component_type == "node" and any(key in low for key in ["node", "cpu", "network", "delay", "loss", "disk"]):
        return 0.85
    if component_type == "pod" and any(key in low for key in ["container", "jvm", "memory", "cpu", "packet", "network"]):
        return 0.75
    if component_type in {"service", "middleware"}:
        # Network issues are common service-level root causes
        # Latency is more common than packet loss for services
        # (services experience backend overloads, not network-layer issues)
        if "latency" in low or "timeout" in low or "delay" in low:
            return 0.90
        if any(key in low for key in ["network", "packet", "loss"]):
            return 0.85
        if any(key in low for key in ["cpu", "jvm", "load"]):
            return 0.60
        if any(key in low for key in ["memory", "oom", "heap"]):
            return 0.55
        if any(key in low for key in ["db", "connection"]):
            return 0.70
    return 0.35


def log_reason_score(reason: str, text: str) -> float:
    """Score log evidence for a reason using weighted keyword frequency.

    Technical solution Section 15.2 Step R2.3:
    - Error/FATAL logs get weight x2
    - Score = hits / (num_keywords_in_family / 2)
    """
    low_reason = reason.lower()
    low_text = text.lower()
    # Count error-level lines for double weighting
    error_count = low_text.count("error") + low_text.count("fatal") + low_text.count("exception")
    keywords = {
        "cpu": ["cpu", "load average", "processor", "throttl", "load", "busy", "fault"],
        "memory": ["memory", "mem", "oom", "outofmemory", "heap", "swap", "gc overhead", "fullgc", "concurrent mode failure"],
        "network": ["network", "timeout", "latency", "delay", "packet", "loss", "retransmit", "retrans", "connection refused", "refused", "slow", "elapsed"],
        "disk": ["disk", "i/o", "io_", "iowait", "filesystem", "throughput", "space", "read", "write"],
        "db": ["jdbc", "sql", "database", "db", "connection", "close", "limit", "pool", "connect"],
        "process": ["restart", "terminated", "killed", "crash", "segfault", "exit", "kill"],
    }
    # Light penalty: "gc" alone is common in Java services and NOT a reliable memory
    # indicator. Only count it at half weight by treating it as a separate low-weight family.
    gc_keywords = ["gc", "cms", "young generation", "minor collection"]
    score = 0.0
    for family, keys in keywords.items():
        if family in low_reason or any(key in low_reason for key in keys):
            hits = sum(1 for key in keys if key in low_text)
            # Error logs get extra weight (technical solution: error weight x2)
            error_bonus = min(error_count * 0.1, 0.3)
            raw_score = hits / max(1, len(keys) / 2)
            score = max(score, min(1.0, raw_score + error_bonus))
    # GC keywords: common in Java services, low weight (0.5x)
    if any(key in low_reason for key in gc_keywords) or "memory" in low_reason:
        gc_hits = sum(1 for key in gc_keywords if key in low_text)
        if gc_hits > 0:
            gc_score = 0.5 * gc_hits / max(1, len(gc_keywords) / 2)
            score = max(score, min(1.0, gc_score))
    return score
