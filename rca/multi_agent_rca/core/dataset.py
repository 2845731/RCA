from __future__ import annotations

import importlib
import re
from pathlib import Path
from typing import Dict, List, Sequence

from rca.multi_agent_rca.core.schema import RCAQuery
from rca.multi_agent_rca.core.time_utils import parse_instruction_time_range


TASK_FIELDS = {
    "task_1": ["root cause occurrence datetime"],
    "task_2": ["root cause reason"],
    "task_3": ["root cause component"],
    "task_4": ["root cause occurrence datetime", "root cause reason"],
    "task_5": ["root cause occurrence datetime", "root cause component"],
    "task_6": ["root cause component", "root cause reason"],
    "task_7": ["root cause component", "root cause occurrence datetime", "root cause reason"],
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def dataset_dir(dataset: str) -> Path:
    return project_root() / "dataset" / Path(dataset)


def telemetry_dir(dataset: str) -> Path:
    return dataset_dir(dataset) / "telemetry"


def _prompt_module_name(dataset: str) -> str:
    if dataset == "Bank":
        return "rca.baseline.rca_agent.prompt.basic_prompt_Bank"
    if dataset == "Telecom":
        return "rca.baseline.rca_agent.prompt.basic_prompt_Telecom"
    if dataset.startswith("Market"):
        return "rca.baseline.rca_agent.prompt.basic_prompt_Market"
    raise ValueError(f"Unsupported dataset: {dataset}")


def _bullet_items(section: str) -> List[str]:
    items = []
    for line in section.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            value = stripped[2:].strip()
            if value and not value.startswith("("):
                items.append(value)
    return items


def candidates_for_dataset(dataset: str) -> Dict[str, List[str]]:
    module = importlib.import_module(_prompt_module_name(dataset))
    cand = getattr(module, "cand", "")
    reason_section = ""
    component_section = ""
    if "POSSIBLE ROOT CAUSE REASONS" in cand:
        after = cand.split("POSSIBLE ROOT CAUSE REASONS:", 1)[1]
        reason_section = after.split("POSSIBLE ROOT CAUSE COMPONENTS:", 1)[0]
    if "POSSIBLE ROOT CAUSE COMPONENTS" in cand:
        component_section = cand.split("POSSIBLE ROOT CAUSE COMPONENTS:", 1)[1]
    return {
        "reasons": _bullet_items(reason_section),
        "components": _bullet_items(component_section),
    }


def infer_target_fields(task_index: str, instruction: str) -> Sequence[str]:
    if task_index in TASK_FIELDS:
        return TASK_FIELDS[task_index]
    fields = []
    lower = str(instruction).lower()
    if "time" in lower or "datetime" in lower or "occurrence" in lower:
        fields.append("root cause occurrence datetime")
    if "component" in lower:
        fields.append("root cause component")
    if "reason" in lower or "why" in lower:
        fields.append("root cause reason")
    return fields or TASK_FIELDS["task_7"]


def infer_failure_count(instruction: str) -> int:
    lower = str(instruction).lower()
    if "single failure" in lower or "one failure" in lower:
        return 1
    m = re.search(r"number of failures(?: recorded)?(?: within this time range)? is (\d+)", lower)
    if not m:
        m = re.search(r"number of failures:\s*(\d+)", lower)
    if m:
        return max(1, min(3, int(m.group(1))))
    if "multiple" in lower or "more than one" in lower:
        return 3
    return 1


def build_query(dataset: str, row_id: int, task_index: str, instruction: str) -> RCAQuery:
    start_time, end_time, start_ts, end_ts = parse_instruction_time_range(instruction)
    candidates = candidates_for_dataset(dataset)
    return RCAQuery(
        dataset=dataset,
        row_id=row_id,
        task_index=task_index,
        instruction=instruction,
        start_time=start_time,
        end_time=end_time,
        start_ts=start_ts,
        end_ts=end_ts,
        target_fields=infer_target_fields(task_index, instruction),
        failure_count=infer_failure_count(instruction),
        candidate_components=candidates["components"],
        candidate_reasons=candidates["reasons"],
    )
