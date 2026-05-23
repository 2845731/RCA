from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from src.config import MemoryConfig
from src.memory.experience_store import ExperienceStore
from src.schemas import SpectralExperience


class PatternEvolver:
    """Self-evolving pattern distillation from experience store.

    Implements the evolution component of Innovation 2: Self-Evolving Causal Graph.

    Evolution mechanism (inspired by EvolveR):
    1. Offline distillation: Extract generalized spectral-propagation rules
       from successful cases. A rule has the form:
       "If component X shows anomaly_type T with spectral pattern P,
        then it is likely the root cause with reason R."
    2. Online adaptation: Adjust rule weights based on new case outcomes.
    3. Pruning: Remove rules with low success rates.
    """

    def __init__(
        self,
        experience_store: ExperienceStore,
        config: Optional[MemoryConfig] = None,
    ) -> None:
        self.store = experience_store
        self.config = config or MemoryConfig()
        self._rules: Dict[str, Dict[str, Any]] = {}

    def evolve(self) -> Dict[str, Any]:
        """Run one evolution cycle: distill, adapt, prune.

        Returns evolution statistics.
        """
        stats = {
            "rules_before": len(self._rules),
            "distilled": 0,
            "pruned": 0,
            "rules_after": 0,
        }

        self._distill()
        stats["distilled"] = len(self._rules) - stats["rules_before"]

        pruned = self._prune()
        stats["pruned"] = pruned
        stats["rules_after"] = len(self._rules)

        return stats

    def get_relevant_rules(
        self,
        component: str,
        anomaly_type: str,
    ) -> List[Dict[str, Any]]:
        """Get evolved rules relevant to a component and anomaly type.

        Returns rules sorted by confidence (success_rate * weight).
        """
        relevant = []
        for rule_key, rule in self._rules.items():
            if rule.get("component") == component or rule.get("anomaly_type") == anomaly_type:
                relevant.append(rule)

        relevant.sort(key=lambda r: r.get("confidence", 0.0), reverse=True)
        return relevant

    def get_rule_for_hypothesis(
        self,
        anomaly_type: str,
        spectral_pattern: Dict[str, float],
    ) -> Optional[Dict[str, Any]]:
        """Get the most relevant evolved rule for a spectral hypothesis.

        Used by the abductive reasoning engine to constrain hypothesis generation.
        """
        best_rule = None
        best_score = 0.0

        for rule_key, rule in self._rules.items():
            if rule.get("anomaly_type") != anomaly_type:
                continue

            pattern_sim = self._pattern_similarity(
                spectral_pattern, rule.get("spectral_pattern_centroid", {})
            )
            score = rule.get("confidence", 0.0) * pattern_sim

            if score > best_score:
                best_score = score
                best_rule = rule

        return best_rule

    def _distill(self) -> None:
        """Distill generalized rules from successful experiences."""
        successful = self.store.get_successful_patterns()

        groups: Dict[Tuple[str, str], List[SpectralExperience]] = {}
        for exp in successful:
            key = (exp.component, exp.anomaly_type)
            groups.setdefault(key, []).append(exp)

        for (component, anomaly_type), experiences in groups.items():
            rule_key = f"{component}::{anomaly_type}"

            patterns = [e.spectral_pattern for e in experiences]
            centroid = self._compute_centroid(patterns)

            reasons = [e.reason for e in experiences]
            reason_counts: Dict[str, int] = {}
            for r in reasons:
                reason_counts[r] = reason_counts.get(r, 0) + 1
            most_common_reason = max(reason_counts, key=reason_counts.get)

            success_count = len(experiences)
            failed_with_same_pattern = sum(
                1 for e in self.store.get_failed_patterns()
                if e.component == component and e.anomaly_type == anomaly_type
            )
            total = success_count + failed_with_same_pattern
            success_rate = success_count / max(total, 1)

            confidence = success_rate * min(1.0, total / 5.0)

            if confidence >= self.config.experience_distill_threshold:
                self._rules[rule_key] = {
                    "component": component,
                    "anomaly_type": anomaly_type,
                    "spectral_pattern_centroid": centroid,
                    "most_common_reason": most_common_reason,
                    "success_rate": round(success_rate, 4),
                    "confidence": round(confidence, 4),
                    "sample_count": total,
                }

    def _prune(self) -> int:
        """Prune rules with low confidence.

        Returns the number of pruned rules.
        """
        to_prune = []
        for rule_key, rule in self._rules.items():
            if rule.get("confidence", 0.0) < self.config.pattern_prune_threshold:
                to_prune.append(rule_key)

        for key in to_prune:
            del self._rules[key]

        return len(to_prune)

    def _compute_centroid(self, patterns: List[Dict[str, float]]) -> Dict[str, float]:
        """Compute the centroid (average) of spectral patterns."""
        if not patterns:
            return {}

        all_keys = set()
        for p in patterns:
            all_keys.update(p.keys())

        centroid: Dict[str, float] = {}
        for key in all_keys:
            values = [p.get(key, 0.0) for p in patterns]
            centroid[key] = round(sum(values) / len(values), 4)

        return centroid

    def _pattern_similarity(
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
