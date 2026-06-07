from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from causalrca_codex.config import AgentLoopConfig

# Module-level LLM response cache: hash(system+user) -> parsed JSON
_LLM_CACHE: Dict[str, Optional[Dict[str, Any]]] = {}


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

    def complete_json(self, system: str, user: str, timeout: int = 20, max_retries: int = 2) -> Optional[Dict[str, Any]]:
        self._load()
        if self._client is None:
            return None

        # Check cache
        cache_key = hashlib.md5((system + "\x00" + user).encode()).hexdigest()
        if cache_key in _LLM_CACHE:
            return _LLM_CACHE[cache_key]

        # Combine system+user into single user message (API doesn't support system role well)
        combined = f"{system}\n\n{user}" if system else user

        import time as _time
        result = None
        for attempt in range(max_retries):
            try:
                response = self._client(
                    messages=[{"role": "user", "content": combined}],
                    timeout=timeout,
                )
                match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
                payload = match.group(1) if match else response
                result = json.loads(payload)
                break
            except Exception:
                if attempt < max_retries - 1:
                    _time.sleep(1)
                continue

        _LLM_CACHE[cache_key] = result
        return result

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
