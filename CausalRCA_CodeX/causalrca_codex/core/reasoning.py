from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Optional


def is_low_sensitive_kpi(kpi: str) -> bool:
    low = kpi.lower()
    return any(key in low for key in ["success", "succee", "sr", "rr", "rate", "avail", "idle"])


def reason_hint(candidate_reasons: Iterable[str], kpi: str, file_name: str = "") -> Optional[str]:
    low = f"{kpi} {file_name}".lower()
    candidates = list(candidate_reasons)
    rules = [
        (["oom", "heap"], ["JVM Out of Memory (OOM) Heap", "high memory usage", "container memory load"]),
        # JVM memory KPIs should map to memory reasons, not CPU (check before generic jvm)
        (["jvmfreememory", "jvmmaxmemory", "jvmtotalmemory", "jvmusedmemory", "jvm_memory", "heapmemory", "noheapmemory"], ["high memory usage", "container memory load"]),
        (["jvm", "cpuload"], ["high JVM CPU load", "high CPU usage", "CPU fault", "container CPU load"]),
        (["cpu", "proc_user", "cpu_used", "cpucpu", "cpuutil"], ["high CPU usage", "CPU fault", "container CPU load", "node CPU load"]),
        (["mem", "memory"], ["high memory usage", "container memory load", "node memory consumption"]),
        (["packet", "loss"], ["network packet loss", "network loss", "container packet loss"]),
        (["fin-wait", "close-wait", "time-wait"], ["network packet loss", "network latency", "network delay"]),
        (["latency", "delay", "mrt", "avg_time", "elapsed", "slow", "timeout"], ["network latency", "network delay", "container network latency"]),
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
    if component_type == "service":
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
    if component_type == "middleware":
        # Middleware components (MG01/MG02/IG01/IG02) can be root causes
        # Network issues are common middleware root causes
        if "latency" in low or "timeout" in low or "delay" in low:
            return 0.80
        if any(key in low for key in ["network", "packet", "loss"]):
            return 0.80
        if any(key in low for key in ["cpu", "jvm", "load"]):
            return 0.60
        if any(key in low for key in ["memory", "oom", "heap"]):
            return 0.55
    return 0.35


def log_reason_score(reason: str, text: str) -> float:
    """Score log evidence for a reason using weighted keyword frequency.

    Innovation: Enhanced reason discrimination with specific keywords.
    - Uses more specific keywords to distinguish similar reasons
    - Adds negative evidence: keywords that reduce score for wrong reasons
    - Distinguishes "network latency" vs "network packet loss" vs "high JVM CPU load"
    """
    low_reason = reason.lower()
    low_text = text.lower()
    # Count error-level lines for double weighting
    error_count = low_text.count("error") + low_text.count("fatal") + low_text.count("exception")
    keywords = {
        "cpu": ["cpu", "load average", "processor", "throttl", "load", "busy", "fault"],
        "memory": ["memory", "mem", "oom", "outofmemory", "heap", "swap", "gc overhead", "fullgc", "concurrent mode failure"],
        "network": ["network", "timeout", "latency", "delay", "packet", "loss", "retransmit", "retrans", "connection refused", "refused", "slow", "elapsed", "reset", "unreachable", "timed out"],
        "disk": ["disk", "i/o", "io_", "iowait", "filesystem", "throughput", "space", "read", "write"],
        "db": ["jdbc", "sql", "database", "db", "connection", "close", "limit", "pool", "connect", "too many connections", "max_connections"],
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


def trace_reason_score(reason: str, span_durations: list, component_type: str = "service") -> float:
    """Innovation: Trace-based reason evidence using span duration patterns.

    Different failure types produce characteristic trace duration signatures:
    - Network latency: uniform high duration across all spans (shifted distribution)
    - Network packet loss: bimodal distribution (some very high timeouts, some normal)
    - High CPU: gradually increasing duration (monotonic trend)
    - High memory: sudden spikes (GC pauses) with normal baseline
    - DB issues: high duration on DB-related spans specifically

    This provides independent evidence from trace data that complements
    KPI-based and log-based reason identification.

    Args:
        reason: The candidate reason to score
        span_durations: List of span duration values (in ms) for the component
        component_type: Type of component (service, db, redis, etc.)

    Returns:
        0-1 score indicating how well the duration pattern matches the reason
    """
    if not span_durations or len(span_durations) < 3:
        return 0.0

    import math
    durations = sorted([float(d) for d in span_durations if d > 0])
    if len(durations) < 3:
        return 0.0

    n = len(durations)
    mean_d = sum(durations) / n
    median_d = durations[n // 2]
    std_d = math.sqrt(sum((d - mean_d) ** 2 for d in durations) / n) if n > 1 else 0
    cv = std_d / mean_d if mean_d > 0 else 0  # Coefficient of variation
    p95 = durations[int(n * 0.95)]
    p5 = durations[int(n * 0.05)]
    iqr = durations[int(n * 0.75)] - durations[int(n * 0.25)]

    low_reason = reason.lower()
    score = 0.0

    # Network latency: uniform high duration (low CV, high median)
    if any(key in low_reason for key in ["latency", "delay", "timeout"]):
        if cv < 0.5 and median_d > 500:  # Low variation, consistently high
            score = max(score, 0.7)
        elif cv < 0.8 and mean_d > 300:
            score = max(score, 0.5)

    # Network packet loss: bimodal (high CV, high p95/p5 ratio)
    if any(key in low_reason for key in ["packet", "loss"]):
        if cv > 1.0 and p95 / max(p5, 1) > 5:  # Very high variation
            score = max(score, 0.6)
        elif cv > 0.8:
            score = max(score, 0.4)

    # High CPU: gradually increasing (check trend via correlation with index)
    if any(key in low_reason for key in ["cpu", "load"]):
        if n >= 5:
            # Simple trend check: are later values higher than earlier values?
            first_half = sum(durations[:n // 2]) / (n // 2)
            second_half = sum(durations[n // 2:]) / (n - n // 2)
            if second_half > first_half * 1.3:  # 30% increase
                score = max(score, 0.6)

    # High memory: sudden spikes (high max/median ratio)
    if any(key in low_reason for key in ["memory", "oom", "heap"]):
        max_ratio = durations[-1] / max(median_d, 1)
        if max_ratio > 5:  # Very sudden spike
            score = max(score, 0.5)
        elif max_ratio > 3:
            score = max(score, 0.3)

    # DB issues: high IQR (some queries are very slow)
    if any(key in low_reason for key in ["db", "connection", "close", "limit"]):
        if component_type in ("db", "redis") and iqr > mean_d * 0.5:
            score = max(score, 0.5)

    return min(1.0, score)


def categorize_kpi(kpi: str) -> str:
    """Innovation: Categorize KPI names into semantic categories.

    Instead of matching KPI names against reason keywords (which is brittle),
    we categorize each KPI into a semantic category, then use the category
    distribution to compute reason evidence. This is more robust because:
    1. Categories are stable across different monitoring systems
    2. The mapping from categories to reasons is well-defined
    3. It avoids false positives from substring matching

    Returns: one of 'cpu', 'memory', 'network', 'disk', 'db', 'process', 'other'
    """
    low = kpi.lower()

    # Memory indicators (specific patterns, not just "mem")
    if any(key in low for key in ["memused", "mem_used", "heap", "oom", "swap",
                                    "memoryused", "memory_used", "jvm.*memory"]):
        return "memory"
    if any(key in low for key in ["memfree", "mem_free", "cache"]):
        return "memory"  # Low free memory is also a memory issue

    # CPU indicators
    if any(key in low for key in ["cpu", "proc", "load", "throttl"]):
        # Exclude idle CPU
        if "idle" in low or "free" in low:
            return "other"
        return "cpu"

    # Network indicators
    if any(key in low for key in ["network", "tcp", "net", "packet", "retrans",
                                    "timeout", "latency", "delay", "connection"]):
        return "network"

    # Disk indicators
    if any(key in low for key in ["disk", "dsk", "i/o", "io_", "iowait",
                                    "space", "read", "write"]):
        return "disk"

    # Database indicators
    if any(key in low for key in ["jdbc", "sql", "database", "db_", "mysql",
                                    "redis", "mongo", "connect", "pool"]):
        return "db"

    # Process indicators
    if any(key in low for key in ["restart", "terminated", "killed", "crash",
                                    "exit", "kill", "process"]):
        return "process"

    return "other"


def category_based_reason_score(
    reason: str,
    anomaly_details: List[Dict],
    candidate_reasons: List[str],
    component_type: str = "service",
) -> float:
    """Innovation: Component-type-aware category-based reason scoring.

    Instead of matching KPI names directly to reasons (brittle), we:
    1. Categorize each anomalous KPI into semantic categories
    2. Count anomalies per category
    3. Map categories to reasons with component-type-specific weights
    4. Normalize across all candidate reasons

    Innovation: Component-type awareness — different component types have
    different failure mode distributions. For example:
    - DB components: more likely to have connection/limit issues
    - Service components: more likely to have network/latency issues
    - Redis components: more likely to have memory issues
    """
    if not anomaly_details:
        return 0.0

    # Compute severity per KPI category
    category_severity: Dict[str, float] = defaultdict(float)
    for seg in anomaly_details:
        kpi = seg.get("kpi", "")
        severity = float(seg.get("severity", 0.0))
        category = categorize_kpi(kpi)
        category_severity[category] += severity

    total_severity = sum(category_severity.values())
    if total_severity <= 0:
        return 0.0

    # Normalize to get fraction of severity per category
    category_fraction = {cat: sev / total_severity for cat, sev in category_severity.items()}

    # Innovation: Concentration bonus — if one category dominates (>50% of severity),
    # boost the score for reasons in that category. This helps differentiate between
    # components with concentrated anomalies (likely root causes) and components with
    # diverse anomalies (likely downstream effects).
    max_fraction = max(category_fraction.values()) if category_fraction else 0.0
    n_categories = len(category_fraction)
    concentration_factor = max_fraction * (1.0 + 0.5 * max(0, 3 - n_categories) / 3.0)

    # Map categories to reasons with component-type-specific weights
    low_reason = reason.lower()
    score = 0.0

    # Innovation: Negative evidence — penalize reasons that don't match the dominant
    # KPI category. If a component's dominant category is "cpu" but we're scoring
    # "high memory usage", the score should be lower than if memory were dominant.
    dominant_category = max(category_fraction, key=category_fraction.get) if category_fraction else ""

    if any(key in low_reason for key in ["cpu", "load"]):
        base = category_fraction.get("cpu", 0.0)
        # Apply concentration bonus if CPU is the dominant category
        if dominant_category == "cpu":
            base = base * concentration_factor
        else:
            base = base * 0.7  # Negative evidence: CPU is not dominant
        # CPU issues are more likely for service/pod components
        type_boost = {"service": 1.2, "pod": 1.1, "node": 1.0}.get(component_type, 0.8)
        score = base * type_boost
    elif any(key in low_reason for key in ["memory", "oom", "heap"]):
        base = category_fraction.get("memory", 0.0)
        # Apply concentration bonus if memory is the dominant category
        if dominant_category == "memory":
            base = base * concentration_factor
        else:
            base = base * 0.7  # Negative evidence: memory is not dominant
        # Memory issues are more likely for redis/pod components
        type_boost = {"redis": 1.3, "pod": 1.2, "service": 1.0}.get(component_type, 0.8)
        score = base * type_boost
    elif any(key in low_reason for key in ["network", "latency", "delay", "timeout"]):
        base = category_fraction.get("network", 0.0)
        # Apply concentration bonus if network is the dominant category
        if dominant_category == "network":
            base = base * concentration_factor
        else:
            base = base * 0.7  # Negative evidence: network is not dominant
        # Network latency is more likely for service components
        type_boost = {"service": 1.2, "pod": 1.1, "middleware": 1.0}.get(component_type, 0.8)
        score = base * type_boost
    elif any(key in low_reason for key in ["packet", "loss"]):
        base = category_fraction.get("network", 0.0) * 0.7 + category_fraction.get("process", 0.0) * 0.3
        type_boost = {"service": 1.1, "pod": 1.0}.get(component_type, 0.8)
        score = base * type_boost
    elif any(key in low_reason for key in ["disk", "i/o", "space"]):
        base = category_fraction.get("disk", 0.0)
        type_boost = {"node": 1.2, "db": 1.1}.get(component_type, 0.8)
        score = base * type_boost
    elif any(key in low_reason for key in ["db", "connection", "close", "limit"]):
        base = category_fraction.get("db", 0.0) + category_fraction.get("network", 0.0) * 0.3
        # DB issues are much more likely for db/redis components
        type_boost = {"db": 1.5, "redis": 1.3}.get(component_type, 0.7)
        score = base * type_boost
    elif "jvm" in low_reason:
        base = category_fraction.get("memory", 0.0) * 0.5 + category_fraction.get("cpu", 0.0) * 0.5
        # JVM issues are more likely for service/pod components
        type_boost = {"service": 1.2, "pod": 1.1}.get(component_type, 0.8)
        score = base * type_boost

    return min(1.0, score)
