from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple


def default_openrca_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass
class AgentLoopConfig:
    """Runtime knobs for the CausalRCA-Flow agent loop."""

    openrca_root: Path = field(default_factory=default_openrca_root)
    dataset_root: Optional[Path] = None
    output_root: Optional[Path] = None

    max_iterations: int = 30
    top_k: int = 4
    quality_alpha: float = 0.6
    low_quality_threshold: float = 0.45

    threshold_percentile: float = 95.0
    low_percentile: float = 5.0
    severity_threshold: float = 0.05
    min_fault_points: int = 2
    beta_min: float = 0.50
    max_candidate_components: int = 30
    time_window_extension_minutes: int = 5
    tau_single: float = 0.80
    allow_single_component_shortcut: bool = True

    expand_mode: str = "direct"
    max_path_depth: int = 6
    lambda_t: float = 300.0
    gamma_time_penalty: float = 0.20
    lambda_call_count: float = 5.0
    lambda_early: float = 300.0

    # Scheme-F 融合权重（因果推理优先：时间序+Granger+相关性为主，类型先验为弱先验）
    # 无LLM: w = 0.35*sA + 0.25*sB + 0.15*sE + 0.25*sD
    # 有LLM(仅reasoning): w = 0.35*sA + 0.25*sB + 0.15*sE + 0.25*sD (边评分不用LLM)
    alpha_time: float = 0.35       # sA 时间序证据权重
    alpha_corr: float = 0.25       # sB 相关性证据权重
    alpha_granger: float = 0.15    # sE Granger因果性证据权重（创新：格兰杰因果检验）
    alpha_llm: float = 0.00        # sC LLM证据权重（边评分不用LLM，仅用reasoning）
    alpha_type_prior: float = 0.25 # sD 类型先验权重（降低：减少组件类型偏置）

    eta_explain: float = 0.15
    eta_source: float = 0.40
    eta_early: float = 0.20
    eta_self_severity: float = 0.25

    delta_score: float = 0.50
    delta_margin: float = 0.30
    delta_graph_quality: float = 0.20

    reason_mu_kpi: float = 0.35
    reason_mu_log: float = 0.15
    reason_mu_llm: float = 0.05
    reason_mu_prior: float = 0.45
    use_llm_reasoning: bool = False
    use_llm_edge_scoring: bool = False

    recovery_budget: Dict[str, int] = field(
        default_factory=lambda: {
            "DataAgent": 2,
            "AssociationAgent": 3,
            "FaultIdentificationAgent": 2,
            "CausalGraphAgent": 3,
            "InterventionAgent": 2,
            "CounterfactualAgent": 2,
        }
    )

    type_prior_table: Dict[Tuple[str, str], float] = field(
        default_factory=lambda: {
            ("db", "db"): 0.40,
            ("db", "redis"): 0.40,
            ("db", "service"): 0.55,
            ("db", "pod"): 0.55,
            ("db", "node"): 0.45,
            ("redis", "db"): 0.40,
            ("redis", "redis"): 0.40,
            ("redis", "service"): 0.55,
            ("redis", "pod"): 0.55,
            ("redis", "node"): 0.45,
            ("service", "db"): 0.50,
            ("service", "redis"): 0.50,
            ("service", "service"): 0.55,
            ("service", "pod"): 0.55,
            ("service", "node"): 0.45,
            ("pod", "db"): 0.50,
            ("pod", "redis"): 0.50,
            ("pod", "service"): 0.55,
            ("pod", "pod"): 0.55,
            ("pod", "node"): 0.45,
            ("node", "db"): 0.60,
            ("node", "redis"): 0.60,
            ("node", "service"): 0.60,
            ("node", "pod"): 0.65,
            ("node", "node"): 0.50,
        }
    )

    def resolved_dataset_root(self) -> Path:
        return Path(self.dataset_root) if self.dataset_root else self.openrca_root / "dataset"

    def resolved_output_root(self) -> Path:
        return Path(self.output_root) if self.output_root else self.openrca_root / "test" / "result"
