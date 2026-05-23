from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple


class CoordinatorState(Enum):
    INIT = "INIT"
    PRECHECK = "PRECHECK"
    DISPATCH = "DISPATCH"
    SYNTHESIZE = "SYNTHESIZE"
    RECHECK = "RECHECK"
    DEBATE = "DEBATE"
    REFLECT = "REFLECT"
    FINALIZE = "FINALIZE"


@dataclass
class RCAQuery:
    dataset: str
    row_id: int
    task_index: str
    instruction: str
    start_time: str
    end_time: str
    start_ts: int
    end_ts: int
    target_fields: Sequence[str]
    failure_count: int
    candidate_components: Sequence[str]
    candidate_reasons: Sequence[str]


@dataclass
class AnomalySegment:
    component: str
    kpi: str
    file_name: str
    start_ts: int
    end_ts: int
    start_time: str
    end_time: str
    direction: str
    severity: float
    max_value: float
    threshold: float
    reason_hint: Optional[str] = None
    evidence_rows: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceReport:
    agent_name: str
    evidence_type: str
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    support: List[str] = field(default_factory=list)
    contradiction: List[str] = field(default_factory=list)
    raw_refs: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Node:
    node_id: str
    node_type: str
    label: str
    score: float = 0.0
    attrs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    source: str
    target: str
    edge_type: str
    weight: float = 0.0
    attrs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceGraph:
    nodes: List[Node] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)
    node_scores: Dict[str, float] = field(default_factory=dict)
    edge_scores: Dict[Tuple[str, str, str], float] = field(default_factory=dict)
    provenance: List[str] = field(default_factory=list)

    def add_node(self, node_id: str, node_type: str, label: str, score: float = 0.0, **attrs: Any) -> None:
        if node_id not in self.node_scores:
            self.nodes.append(Node(node_id=node_id, node_type=node_type, label=label, score=score, attrs=attrs))
        self.node_scores[node_id] = self.node_scores.get(node_id, 0.0) + score

    def add_edge(self, source: str, target: str, edge_type: str, weight: float = 0.0, **attrs: Any) -> None:
        key = (source, target, edge_type)
        self.edges.append(Edge(source=source, target=target, edge_type=edge_type, weight=weight, attrs=attrs))
        self.edge_scores[key] = self.edge_scores.get(key, 0.0) + weight

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [asdict(n) for n in self.nodes],
            "edges": [asdict(e) for e in self.edges],
            "node_scores": self.node_scores,
            "edge_scores": {f"{s}|{t}|{k}": v for (s, t, k), v in self.edge_scores.items()},
            "provenance": self.provenance,
        }


@dataclass
class SamplerReport:
    selected_trace_ids: List[str] = field(default_factory=list)
    selected_log_templates: List[str] = field(default_factory=list)
    trace_pattern_coverage: float = 0.0
    shannon_entropy: float = 0.0
    anomaly_ratio: float = 0.0
    input_rows: int = 0
    sampled_rows: int = 0
    elapsed_seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RCAResult:
    prediction_json: str
    ranked_candidates: List[Dict[str, Any]]
    evidence_chain: List[Dict[str, Any]]
    graph: EvidenceGraph
    trajectory: List[Dict[str, Any]]
    cost: Dict[str, Any]
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prediction_json": self.prediction_json,
            "ranked_candidates": self.ranked_candidates,
            "evidence_chain": self.evidence_chain,
            "graph": self.graph.to_dict(),
            "trajectory": self.trajectory,
            "cost": self.cost,
            "diagnostics": self.diagnostics,
        }
