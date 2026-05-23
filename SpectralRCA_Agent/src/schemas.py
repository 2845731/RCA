from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple


class SpectralAnomalyType(Enum):
    SLOW_TREND = "slow_trend_anomaly"
    FAST_BURST = "fast_burst_or_jitter_anomaly"
    PERIODIC_OSCILLATION = "periodic_oscillation_anomaly"
    MIXED_SPECTRAL = "mixed_spectral_anomaly"
    NO_STRONG_SPECTRAL = "no_strong_spectral_anomaly"


class CoordinatorState(Enum):
    INIT = "INIT"
    HYPOTHESIZE = "HYPOTHESIZE"
    EVIDENCE_COLLECT = "EVIDENCE_COLLECT"
    SPECTRAL_VALIDATE = "SPECTRAL_VALIDATE"
    GRAPH_REFINE = "GRAPH_REFINE"
    ABDUCTIVE_REASON = "ABDUCTIVE_REASON"
    BACKTRACK = "BACKTRACK"
    SYNTHESIZE = "SYNTHESIZE"
    REFLECT = "REFLECT"
    FINALIZE = "FINALIZE"


class BeliefState(Enum):
    EMPTY = "empty"
    HYPOTHESIZED = "hypothesized"
    EVIDENCE_SUPPORTED = "evidence_supported"
    EVIDENCE_CONTRADICTED = "evidence_contradicted"
    SPECTRAL_CONFIRMED = "spectral_confirmed"
    SPECTRAL_REJECTED = "spectral_rejected"
    VALIDATED = "validated"
    REJECTED = "rejected"


@dataclass
class MetricAnomalyEvidence:
    node_id: str
    traditional_score: float
    spectral_score: float
    final_anomaly_score: float
    anomaly_type: str
    max_abs_robust_z: float
    max_change_rate: float
    max_deviation_score: float
    total_energy: float
    spectral_energy_z: float
    low_ratio: float
    mid_ratio: float
    high_ratio: float
    dominant_freq_index: int
    dominant_freq_energy_ratio: float
    spectral_entropy: float
    quality_flag: str
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EdgeEvidence:
    source: str
    target: str
    prior_weight: float
    node_anomaly_factor: float
    best_lag: int
    lag_corr: float
    spectral_shape_similarity: float
    dominant_freq_match: float
    phase_lag_consistency: float
    graph_consistency: float
    final_edge_weight: float
    keep_edge: bool
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RootCauseCandidate:
    node_id: str
    root_score: float
    anomaly_score: float
    out_evidence: float
    in_evidence: float
    onset_time: Optional[str] = None
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


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
    spectral_anomaly_type: Optional[str] = None
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
class RCAResult:
    prediction_json: str
    ranked_candidates: List[Dict[str, Any]]
    evidence_chain: List[Dict[str, Any]]
    graph: EvidenceGraph
    anomaly_evidence: List[MetricAnomalyEvidence]
    edge_evidence: List[EdgeEvidence]
    root_cause_ranking: List[RootCauseCandidate]
    trajectory: List[Dict[str, Any]]
    cost: Dict[str, Any]
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prediction_json": self.prediction_json,
            "ranked_candidates": self.ranked_candidates,
            "evidence_chain": self.evidence_chain,
            "graph": self.graph.to_dict(),
            "anomaly_evidence": [e.to_dict() for e in self.anomaly_evidence],
            "edge_evidence": [e.to_dict() for e in self.edge_evidence],
            "root_cause_ranking": [r.to_dict() for r in self.root_cause_ranking],
            "trajectory": self.trajectory,
            "cost": self.cost,
            "diagnostics": self.diagnostics,
        }


@dataclass
class AbductiveHypothesis:
    component: str
    reason: Optional[str]
    spectral_anomaly_type: Optional[str]
    belief_state: BeliefState
    evidence_for: List[str]
    evidence_against: List[str]
    confidence: float
    iteration: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component": self.component,
            "reason": self.reason,
            "spectral_anomaly_type": self.spectral_anomaly_type,
            "belief_state": self.belief_state.value,
            "evidence_for": self.evidence_for,
            "evidence_against": self.evidence_against,
            "confidence": self.confidence,
            "iteration": self.iteration,
        }


@dataclass
class SpectralExperience:
    component: str
    kpi: str
    anomaly_type: str
    spectral_pattern: Dict[str, float]
    reason: str
    success: bool
    timestamp: str
    case_id: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
