from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

from src.config import MemoryConfig
from src.schemas import SpectralExperience


class ExperienceStore:
    """Persistent store for spectral experiences from past RCA cases.

    Implements the memory component of Innovation 2: Self-Evolving Causal Graph.
    Stores successful and failed spectral patterns to guide future diagnosis.

    Experience format:
        - component: The root cause component
        - kpi: The anomalous KPI
        - anomaly_type: Spectral anomaly type (slow_trend, fast_burst, etc.)
        - spectral_pattern: Dict of spectral features
        - reason: The root cause reason
        - success: Whether this pattern led to correct diagnosis
        - timestamp: When this case occurred
        - case_id: Unique identifier for the case
    """

    def __init__(self, config: Optional[MemoryConfig] = None) -> None:
        self.config = config or MemoryConfig()
        self._experiences: List[SpectralExperience] = []
        self._index_by_component: Dict[str, List[int]] = {}
        self._index_by_anomaly_type: Dict[str, List[int]] = {}
        self._index_by_reason: Dict[str, List[int]] = {}

        if self.config.memory_dir:
            self._load_from_disk()

    def add(self, experience: SpectralExperience) -> None:
        """Add a new experience to the store."""
        if self.config.frozen:
            return

        idx = len(self._experiences)
        self._experiences.append(experience)

        self._index_by_component.setdefault(experience.component, []).append(idx)
        self._index_by_anomaly_type.setdefault(experience.anomaly_type, []).append(idx)
        self._index_by_reason.setdefault(experience.reason, []).append(idx)

        if self.config.memory_dir:
            self._save_to_disk()

    def add_batch(self, experiences: List[SpectralExperience]) -> None:
        """Add multiple experiences at once."""
        for exp in experiences:
            self.add(exp)

    def retrieve_by_component(
        self,
        component: str,
        max_results: Optional[int] = None,
    ) -> List[SpectralExperience]:
        """Retrieve experiences for a specific component."""
        max_results = max_results or self.config.max_retrieved_cases
        indices = self._index_by_component.get(component, [])
        return [self._experiences[i] for i in indices[:max_results]]

    def retrieve_by_anomaly_type(
        self,
        anomaly_type: str,
        max_results: Optional[int] = None,
    ) -> List[SpectralExperience]:
        """Retrieve experiences with a specific spectral anomaly type."""
        max_results = max_results or self.config.max_retrieved_cases
        indices = self._index_by_anomaly_type.get(anomaly_type, [])
        return [self._experiences[i] for i in indices[:max_results]]

    def retrieve_similar(
        self,
        component: str,
        anomaly_type: str,
        spectral_pattern: Dict[str, float],
        max_results: Optional[int] = None,
    ) -> List[Tuple[SpectralExperience, float]]:
        """Retrieve similar experiences based on component, type, and pattern similarity.

        Returns list of (experience, similarity_score) tuples sorted by similarity.
        """
        max_results = max_results or self.config.max_retrieved_cases

        candidates = []
        comp_indices = set(self._index_by_component.get(component, []))
        type_indices = set(self._index_by_anomaly_type.get(anomaly_type, []))

        relevant_indices = comp_indices | type_indices
        if not relevant_indices:
            relevant_indices = set(range(len(self._experiences)))

        for idx in relevant_indices:
            if idx >= len(self._experiences):
                continue
            exp = self._experiences[idx]
            similarity = self._compute_pattern_similarity(spectral_pattern, exp.spectral_pattern)
            if exp.component == component:
                similarity += 0.2
            if exp.anomaly_type == anomaly_type:
                similarity += 0.2
            candidates.append((exp, min(1.0, similarity)))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:max_results]

    def get_successful_patterns(self) -> List[SpectralExperience]:
        """Get all experiences that led to successful diagnosis."""
        return [e for e in self._experiences if e.success]

    def get_failed_patterns(self) -> List[SpectralExperience]:
        """Get all experiences that led to incorrect diagnosis."""
        return [e for e in self._experiences if not e.success]

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the experience store."""
        from typing import Any
        total = len(self._experiences)
        successful = sum(1 for e in self._experiences if e.success)
        type_dist: Dict[str, int] = {}
        for e in self._experiences:
            type_dist[e.anomaly_type] = type_dist.get(e.anomaly_type, 0) + 1

        return {
            "total_experiences": total,
            "successful": successful,
            "failed": total - successful,
            "success_rate": successful / max(total, 1),
            "anomaly_type_distribution": type_dist,
            "unique_components": len(self._index_by_component),
            "unique_reasons": len(self._index_by_reason),
        }

    def _compute_pattern_similarity(
        self,
        pattern_a: Dict[str, float],
        pattern_b: Dict[str, float],
    ) -> float:
        """Compute cosine similarity between two spectral patterns."""
        import numpy as np

        keys = set(pattern_a.keys()) | set(pattern_b.keys())
        if not keys:
            return 0.0

        vec_a = np.array([pattern_a.get(k, 0.0) for k in keys])
        vec_b = np.array([pattern_b.get(k, 0.0) for k in keys])

        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        if norm_a < 1e-12 or norm_b < 1e-12:
            return 0.0

        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

    def _save_to_disk(self) -> None:
        """Save experiences to disk."""
        if not self.config.memory_dir:
            return

        os.makedirs(self.config.memory_dir, exist_ok=True)
        path = os.path.join(self.config.memory_dir, "experiences.json")
        data = [exp.to_dict() for exp in self._experiences]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load_from_disk(self) -> None:
        """Load experiences from disk."""
        if not self.config.memory_dir:
            return

        path = os.path.join(self.config.memory_dir, "experiences.json")
        if not os.path.exists(path):
            return

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            exp = SpectralExperience(
                component=item.get("component", ""),
                kpi=item.get("kpi", ""),
                anomaly_type=item.get("anomaly_type", ""),
                spectral_pattern=item.get("spectral_pattern", {}),
                reason=item.get("reason", ""),
                success=item.get("success", False),
                timestamp=item.get("timestamp", ""),
                case_id=item.get("case_id", ""),
            )
            idx = len(self._experiences)
            self._experiences.append(exp)
            self._index_by_component.setdefault(exp.component, []).append(idx)
            self._index_by_anomaly_type.setdefault(exp.anomaly_type, []).append(idx)
            self._index_by_reason.setdefault(exp.reason, []).append(idx)
