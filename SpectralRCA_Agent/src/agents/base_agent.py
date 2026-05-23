from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from src.schemas import EvidenceReport, RCAQuery


class BaseAgent(ABC):
    """Abstract base class for all SpectralRCA agents.

    Each agent receives a query and accumulated evidence, then produces
    an EvidenceReport with its findings.
    """

    def __init__(self, name: str, config: Optional[Any] = None) -> None:
        self.name = name
        self.config = config

    @abstractmethod
    def run(
        self,
        query: RCAQuery,
        prior_evidence: Optional[List[EvidenceReport]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> EvidenceReport:
        """Execute the agent's analysis and return an evidence report.

        Args:
            query: The RCA query to analyze.
            prior_evidence: Evidence from previously executed agents.
            context: Additional context (e.g., data loader, memory store).

        Returns:
            EvidenceReport with this agent's findings.
        """
        ...

    def describe(self) -> str:
        """Return a human-readable description of this agent."""
        return f"{self.__class__.__name__}(name={self.name})"
