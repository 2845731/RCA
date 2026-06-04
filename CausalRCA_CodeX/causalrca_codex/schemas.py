from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


TASK_FIELDS: Dict[str, List[str]] = {
    "task_1": ["root cause occurrence datetime"],
    "task_2": ["root cause reason"],
    "task_3": ["root cause component"],
    "task_4": ["root cause occurrence datetime", "root cause reason"],
    "task_5": ["root cause occurrence datetime", "root cause component"],
    "task_6": ["root cause component", "root cause reason"],
    "task_7": [
        "root cause occurrence datetime",
        "root cause component",
        "root cause reason",
    ],
}


@dataclass
class RCAQuery:
    dataset: str
    row_id: int
    task_index: str
    instruction: str
    scoring_points: str
    start_time: str
    end_time: str
    start_ts: int
    end_ts: int
    target_fields: Sequence[str]
    failure_count: int
    candidate_components: Sequence[str]
    candidate_reasons: Sequence[str]


@dataclass
class TelemetryFrame:
    path: str
    kind: str
    file_name: str
    rows: int
    timestamp_col: Optional[str]
    data: Any


@dataclass
class MetricSeries:
    component: str
    raw_component: str
    component_type: str
    kpi: str
    file_name: str
    full: Any
    window: Any
    threshold_high: float
    threshold_low: float
    median: float
    scale: float


@dataclass
class AnomalySegment:
    component: str
    component_type: str
    kpi: str
    file_name: str
    start_ts: int
    end_ts: int
    start_time: str
    end_time: str
    direction: str
    peak_value: float
    threshold: float
    max_deviation: float
    severity: float
    reason_hint: Optional[str] = None
    points: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GraphEdge:
    source: str
    target: str
    weight: float
    call_count: int = 0
    scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WeightedCausalGraph:
    nodes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    edges: List[GraphEdge] = field(default_factory=list)

    def add_node(self, node: str, **attrs: Any) -> None:
        current = self.nodes.setdefault(node, {})
        current.update({k: v for k, v in attrs.items() if v is not None})

    def add_edge(self, source: str, target: str, weight: float, **attrs: Any) -> None:
        self.add_node(source)
        self.add_node(target)
        self.edges.append(GraphEdge(source=source, target=target, weight=weight, **attrs))

    def outgoing(self, node: str) -> List[GraphEdge]:
        return [edge for edge in self.edges if edge.source == node]

    def incoming(self, node: str) -> List[GraphEdge]:
        return [edge for edge in self.edges if edge.target == node]

    def has_node(self, node: str) -> bool:
        return node in self.nodes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": self.nodes,
            "edges": [edge.to_dict() for edge in self.edges],
        }


@dataclass
class AgentResult:
    agent_name: str
    status: str
    result: Dict[str, Any]
    self_assessed_quality: float
    warnings: List[str] = field(default_factory=list)
    suggested_next_action: str = "continue"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RootCausePrediction:
    component: str
    occurrence_time: str
    reason: str
    scores: Mapping[str, float] = field(default_factory=dict)
    explanation: str = ""
    causes: Optional[List[Mapping[str, str]]] = None

    def to_opencra_json(self, target_fields: Sequence[str]) -> str:
        causes = list(self.causes or [])
        if not causes:
            causes = [{"component": self.component, "time": self.occurrence_time, "reason": self.reason}]

        items = []
        for idx, cause in enumerate(causes, start=1):
            fields = []
            if "root cause occurrence datetime" in target_fields:
                fields.append(f'"root cause occurrence datetime": "{cause.get("time", "")}"')
            if "root cause component" in target_fields:
                fields.append(f'"root cause component": "{cause.get("component", "")}"')
            if "root cause reason" in target_fields:
                fields.append(f'"root cause reason": "{cause.get("reason", "")}"')
            items.append(f'    "{idx}": {{\n        ' + ",\n        ".join(fields) + "\n    }")
        return "{\n" + ",\n".join(items) + "\n}"


GroundTruth = Dict[str, Any]
CallCounts = Dict[Tuple[str, str], int]
