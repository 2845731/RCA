from __future__ import annotations

import json
from difflib import get_close_matches
from typing import Any, Dict, Iterable, List

from rca.multi_agent_rca.core.schema import RCAQuery


class VerifierAgent:
    """Validate final output against OpenRCA answer constraints."""

    def run(self, query: RCAQuery, prediction_json: str, ranked_candidates: List[Dict[str, Any]]) -> str:
        try:
            parsed = json.loads(prediction_json)
        except Exception:
            parsed = {}

        fixed: Dict[str, Dict[str, str]] = {}
        count = max(1, query.failure_count)
        for idx in range(1, count + 1):
            item = parsed.get(str(idx), {}) if isinstance(parsed, dict) else {}
            candidate = ranked_candidates[min(idx - 1, len(ranked_candidates) - 1)] if ranked_candidates else {}
            out: Dict[str, str] = {}
            if "root cause occurrence datetime" in query.target_fields:
                out["root cause occurrence datetime"] = str(
                    item.get("root cause occurrence datetime") or candidate.get("occurrence_time") or query.start_time
                )
            if "root cause component" in query.target_fields:
                component = str(item.get("root cause component") or candidate.get("component") or "")
                out["root cause component"] = self._coerce(component, query.candidate_components)
            if "root cause reason" in query.target_fields:
                reason = str(item.get("root cause reason") or candidate.get("reason") or "")
                out["root cause reason"] = self._coerce(reason, query.candidate_reasons)
            fixed[str(idx)] = out
        return json.dumps(fixed, ensure_ascii=False, indent=4)

    def _coerce(self, value: str, candidates: Iterable[str]) -> str:
        cand = list(candidates)
        if value in cand:
            return value
        if not cand:
            return value
        matches = get_close_matches(value, cand, n=1, cutoff=0.3)
        return matches[0] if matches else cand[0]
