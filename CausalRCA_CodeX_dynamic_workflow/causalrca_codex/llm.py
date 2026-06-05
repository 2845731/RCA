from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from causalrca_codex.config import AgentLoopConfig


class LLMClient:
    """Thin optional adapter over the repository's existing rca.api_router."""

    def __init__(self, config: AgentLoopConfig) -> None:
        self.config = config
        self._client = None

    def available(self) -> bool:
        try:
            self._load()
            return self._client is not None
        except Exception:
            return False

    def complete_json(self, system: str, user: str) -> Optional[Dict[str, Any]]:
        self._load()
        if self._client is None:
            return None
        response = self._client(messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
        payload = match.group(1) if match else response
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return None

    def _load(self) -> None:
        if self._client is not None:
            return
        root = str(self.config.openrca_root)
        if root not in sys.path:
            sys.path.insert(0, root)
        try:
            from rca.api_router import get_chat_completion

            self._client = get_chat_completion
        except Exception:
            self._client = None
