from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from rca.multi_agent_rca.core.time_utils import normalize_epoch


TIME_COLUMNS = ("timestamp", "startTime")


def read_csv_safe(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def find_time_column(df: pd.DataFrame) -> Optional[str]:
    for col in TIME_COLUMNS:
        if col in df.columns:
            return col
    return None


def normalize_time_column(df: pd.DataFrame) -> Optional[str]:
    col = find_time_column(df)
    if col is None:
        return None
    values = pd.to_numeric(df[col], errors="coerce")
    non_null = values.dropna()
    if not non_null.empty and non_null.abs().median() >= 10_000_000_000:
        values = (values // 1000)
    df[col] = values
    df.dropna(subset=[col], inplace=True)
    df[col] = df[col].astype(int)
    return col


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
