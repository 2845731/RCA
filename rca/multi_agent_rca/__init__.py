"""Multi-agent RCA research prototype for OpenRCA.

The package is intentionally runnable without LLM API keys. It implements the
evidence-producing agents, a MACAA-style coordinator state machine, semantic
sampling, memory hooks, and a lightweight meta-causal graph layer.
"""

from rca.multi_agent_rca.coordinator import Coordinator, CoordinatorConfig

__all__ = ["Coordinator", "CoordinatorConfig"]
