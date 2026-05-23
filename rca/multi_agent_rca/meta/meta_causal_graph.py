from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple


@dataclass
class MetaCausalGraph:
    """Reusable metadata-level causal priors.

    This is a lightweight implementation of the MetaRCA idea: the online RCA
    agents instantiate these priors on concrete components and KPIs. The graph
    can later be evolved from historical incident reports or successful
    trajectories without changing the online interface.
    """

    reason_priors: Dict[str, Dict[str, float]] = field(default_factory=dict)

    @classmethod
    def default(cls, candidate_reasons: Iterable[str]) -> "MetaCausalGraph":
        reasons = set(candidate_reasons)
        priors: Dict[str, Dict[str, float]] = defaultdict(dict)

        def add(metric_type: str, reason: str, weight: float) -> None:
            if reason in reasons:
                priors[metric_type][reason] = weight

        for reason in ["high CPU usage", "CPU fault", "high JVM CPU load"]:
            add("cpu", reason, 0.9)
        for reason in ["high memory usage", "JVM Out of Memory (OOM) Heap"]:
            add("memory", reason, 0.9)
        for reason in ["network latency", "network delay"]:
            add("latency", reason, 0.8)
        for reason in ["network packet loss", "network loss"]:
            add("packet", reason, 0.8)
        add("connection", "db connection limit", 0.8)
        add("connection", "db close", 0.5)
        for reason in ["high disk I/O read usage", "high disk space usage"]:
            add("disk", reason, 0.8)
        return cls(reason_priors=dict(priors))

    def metric_type(self, kpi: str) -> str:
        low = str(kpi).lower()
        if any(x in low for x in ["cpu", "cpuload", "proc_user"]):
            return "cpu"
        if any(x in low for x in ["mem", "memory", "heap", "oom"]):
            return "memory"
        if any(x in low for x in ["packet", "loss", "fin-wait", "close-wait"]):
            return "packet"
        if any(x in low for x in ["latency", "delay", "mrt", "time", "tnsping", "avg_time"]):
            return "latency"
        if any(x in low for x in ["connect", "sess"]):
            return "connection"
        if any(x in low for x in ["disk", "io", "iowait", "read", "write"]):
            return "disk"
        return "unknown"

    def reason_scores_for_kpi(self, kpi: str) -> Dict[str, float]:
        return dict(self.reason_priors.get(self.metric_type(kpi), {}))

    def instantiate_edges(self, component: str, kpi: str, severity: float) -> List[Tuple[str, str, float]]:
        edges = []
        for reason, prior in self.reason_scores_for_kpi(kpi).items():
            edges.append((f"metric:{component}:{kpi}", f"reason:{reason}", prior * severity))
        return edges
