from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import pandas as pd


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if hasattr(value, "to_dict") and not isinstance(value, pd.DataFrame):
        try:
            return to_jsonable(value.to_dict())
        except TypeError:
            pass
    if isinstance(value, pd.DataFrame):
        return {
            "type": "DataFrame",
            "rows": len(value),
            "columns": list(value.columns),
            "preview": value.head(20).to_dict("records"),
        }
    if isinstance(value, pd.Series):
        return value.head(20).to_list()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value
