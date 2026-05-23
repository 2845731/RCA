from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class AnomalyConfig:
    traditional_weight_threshold: float = 0.20
    traditional_weight_zscore: float = 0.35
    traditional_weight_change_rate: float = 0.25
    traditional_weight_max_deviation: float = 0.20
    spectral_weight_energy: float = 0.40
    spectral_weight_low_shift: float = 0.20
    spectral_weight_high_shift: float = 0.20
    spectral_weight_periodic: float = 0.10
    spectral_weight_entropy: float = 0.10
    final_weight_traditional: float = 0.6
    final_weight_spectral: float = 0.4
    high_confidence_threshold: float = 0.85
    candidate_threshold: float = 0.70
    spectral_energy_z_threshold: float = 3.0
    spectral_ratio_z_threshold: float = 2.5
    eps: float = 1e-8


@dataclass
class GraphConfig:
    edge_spectral_weight_shape: float = 0.45
    edge_spectral_weight_freq: float = 0.20
    edge_spectral_weight_phase: float = 0.20
    edge_spectral_weight_lag: float = 0.15
    pruning_threshold: float = 0.30
    max_lag: Optional[int] = None
    graph_consistency_eta: float = 1.0
    freq_tolerance: int = 1
    min_phase_lag_length: int = 16


@dataclass
class MemoryConfig:
    enable_memory: bool = True
    memory_dir: Optional[str] = None
    frozen: bool = True
    max_retrieved_cases: int = 3
    experience_distill_threshold: float = 0.7
    pattern_prune_threshold: float = 0.3


@dataclass
class ReasoningConfig:
    max_abductive_iterations: int = 3
    backtrack_threshold: float = 0.3
    hypothesis_top_k: int = 5
    enable_spectral_belief: bool = True
    enable_backtrack: bool = True


@dataclass
class PipelineConfig:
    resample_interval: str = "60s"
    missing_threshold: float = 0.30
    min_spectral_length: int = 8
    baseline_window_multiplier: int = 3


@dataclass
class LLMConfig:
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: str = "gpt-4"
    temperature: float = 0.1
    max_tokens: int = 1024
    enable_llm: bool = True


@dataclass
class SpectralRCAConfig:
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    dataset_dir: Optional[str] = None

    enable_traditional_anomaly: bool = True
    enable_spectral_anomaly: bool = True
    enable_lag_corr: bool = True
    enable_spectral_shape: bool = True
    enable_dominant_freq_match: bool = True
    enable_phase_lag: bool = True
    enable_graph_consistency: bool = True
    enable_memory: bool = True
    enable_abductive: bool = True
