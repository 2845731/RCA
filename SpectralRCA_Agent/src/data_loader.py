from __future__ import annotations

import os
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from src.schemas import RCAQuery


DATASET_DIRS = {
    "Bank": "Bank",
    "Telecom": "Telecom",
}


def _parse_time_from_instruction(instruction: str) -> Optional[Tuple[str, str]]:
    """Extract incident time range from query instruction text.

    Supports patterns like:
    - 'March 4, 2021, within the time range of 14:30 to 15:00'
    - 'April 11, 2020, from 00:00 to 00:30'
    - 'between 18:00 and 18:30'
    """
    patterns = [
        r"(?:On|During|Within)\s+(?:the\s+)?(?:specified\s+)?(?:time\s+range\s+)?(?:of\s+)?"
        r"(\w+\s+\d{1,2},?\s+\d{4}),?\s+(?:within\s+the\s+time\s+range\s+of\s+)?"
        r"(?:from\s+)?(\d{1,2}:\d{2})\s+(?:to|and)\s+(\d{1,2}:\d{2})",
        r"(\w+\s+\d{1,2},?\s+\d{4}),?\s+(?:from\s+)?(\d{1,2}:\d{2})\s+(?:to|and)\s+(\d{1,2}:\d{2})",
        r"between\s+(\d{1,2}:\d{2})\s+and\s+(\d{1,2}:\d{2})",
    ]

    for pat in patterns:
        m = re.search(pat, instruction, re.IGNORECASE)
        if m:
            groups = m.groups()
            if len(groups) == 3:
                date_str, start_time, end_time = groups
                try:
                    dt = datetime.strptime(date_str.strip(), "%B %d, %Y")
                except ValueError:
                    try:
                        dt = datetime.strptime(date_str.strip(), "%B %d %Y")
                    except ValueError:
                        continue
                start_dt = dt.replace(hour=int(start_time.split(":")[0]), minute=int(start_time.split(":")[1]))
                end_dt = dt.replace(hour=int(end_time.split(":")[0]), minute=int(end_time.split(":")[1]))
                return start_dt.strftime("%Y-%m-%d %H:%M:%S"), end_dt.strftime("%Y-%m-%d %H:%M:%S")
            elif len(groups) == 2:
                start_time, end_time = groups
                return None, None
    return None


def _parse_date_folder(instruction: str) -> Optional[str]:
    """Extract date folder name from instruction (e.g., '2021_03_04')."""
    m = re.search(r"(\w+\s+\d{1,2},?\s+\d{4})", instruction)
    if m:
        date_str = m.group(1).strip()
        for fmt in ["%B %d, %Y", "%B %d %Y"]:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.strftime("%Y_%m_%d")
            except ValueError:
                continue
    return None


def _extract_task_type(instruction: str) -> str:
    """Determine the task type from instruction text."""
    inst_lower = instruction.lower()
    has_component = "component" in inst_lower
    has_reason = "reason" in inst_lower
    has_time = "time" in inst_lower or "datetime" in inst_lower or "occurrence time" in inst_lower

    if has_component and has_reason and has_time:
        return "full"
    elif has_component and has_reason:
        return "component_reason"
    elif has_component and has_time:
        return "component_time"
    elif has_reason and has_time:
        return "reason_time"
    elif has_component:
        return "component"
    elif has_reason:
        return "reason"
    elif has_time:
        return "time"
    return "full"


class DataLoader:
    """Load and parse OpenRCA datasets (Bank, Telecom).

    Handles different CSV schemas across datasets and provides
    unified access to metric, trace, and log data.
    """

    def __init__(self, dataset_dir: str) -> None:
        self.dataset_dir = dataset_dir
        self._query_cache: Dict[str, pd.DataFrame] = {}
        self._record_cache: Dict[str, pd.DataFrame] = {}

    def list_datasets(self) -> List[str]:
        """List available dataset names."""
        datasets = []
        for name in DATASET_DIRS:
            path = os.path.join(self.dataset_dir, DATASET_DIRS[name])
            if os.path.isdir(path):
                datasets.append(name)
        return datasets

    def load_queries(self, dataset: str) -> pd.DataFrame:
        """Load query.csv for a dataset."""
        if dataset in self._query_cache:
            return self._query_cache[dataset]
        path = os.path.join(self.dataset_dir, DATASET_DIRS[dataset], "query.csv")
        df = pd.read_csv(path)
        self._query_cache[dataset] = df
        return df

    def load_records(self, dataset: str) -> pd.DataFrame:
        """Load record.csv (ground truth) for a dataset."""
        if dataset in self._record_cache:
            return self._record_cache[dataset]
        path = os.path.join(self.dataset_dir, DATASET_DIRS[dataset], "record.csv")
        df = pd.read_csv(path)
        self._record_cache[dataset] = df
        return df

    def load_metric_container(self, dataset: str, date_folder: str) -> pd.DataFrame:
        """Load metric_container.csv for a specific date."""
        path = os.path.join(
            self.dataset_dir, DATASET_DIRS[dataset],
            "telemetry", date_folder, "metric", "metric_container.csv",
        )
        if not os.path.exists(path):
            return pd.DataFrame()
        return pd.read_csv(path)

    def load_metric_app(self, dataset: str, date_folder: str) -> pd.DataFrame:
        """Load metric_app.csv for a specific date."""
        path = os.path.join(
            self.dataset_dir, DATASET_DIRS[dataset],
            "telemetry", date_folder, "metric", "metric_app.csv",
        )
        if not os.path.exists(path):
            return pd.DataFrame()
        return pd.read_csv(path)

    def load_all_metrics(self, dataset: str, date_folder: str) -> pd.DataFrame:
        """Load all metric files and unify them into a single DataFrame.

        Returns DataFrame with columns: [timestamp, node_id, kpi_name, value]
        """
        all_dfs = []

        container_df = self.load_metric_container(dataset, date_folder)
        if not container_df.empty:
            if dataset == "Bank":
                container_df = container_df.rename(columns={"cmdb_id": "node_id", "kpi_name": "kpi_name"})
            elif dataset == "Telecom":
                container_df = container_df.rename(columns={"cmdb_id": "node_id", "name": "kpi_name"})
                if "itemid" in container_df.columns:
                    container_df = container_df.drop(columns=["itemid", "bomc_id"], errors="ignore")
            if "node_id" in container_df.columns and "kpi_name" in container_df.columns:
                all_dfs.append(container_df[["timestamp", "node_id", "kpi_name", "value"]].copy())

        app_df = self.load_metric_app(dataset, date_folder)
        if not app_df.empty:
            if dataset == "Bank":
                for kpi in ["rr", "sr", "cnt", "mrt"]:
                    if kpi in app_df.columns:
                        sub = app_df[["timestamp", "tc"]].copy()
                        sub = sub.rename(columns={"tc": "node_id"})
                        sub["kpi_name"] = kpi
                        sub["value"] = app_df[kpi]
                        all_dfs.append(sub[["timestamp", "node_id", "kpi_name", "value"]])
            elif dataset == "Telecom":
                if "serviceName" in app_df.columns:
                    for kpi in ["avg_time", "num", "succee_num", "succee_rate"]:
                        if kpi in app_df.columns:
                            sub = app_df[["startTime", "serviceName"]].copy()
                            sub = sub.rename(columns={"startTime": "timestamp", "serviceName": "node_id"})
                            sub["kpi_name"] = kpi
                            sub["value"] = app_df[kpi]
                            all_dfs.append(sub[["timestamp", "node_id", "kpi_name", "value"]])

        for extra_name in ["metric_middleware", "metric_node", "metric_service"]:
            path = os.path.join(
                self.dataset_dir, DATASET_DIRS[dataset],
                "telemetry", date_folder, "metric", f"{extra_name}.csv",
            )
            if os.path.exists(path):
                extra_df = pd.read_csv(path)
                if dataset == "Telecom":
                    rename_map = {}
                    if "cmdb_id" in extra_df.columns:
                        rename_map["cmdb_id"] = "node_id"
                    if "name" in extra_df.columns:
                        rename_map["name"] = "kpi_name"
                    extra_df = extra_df.rename(columns=rename_map)
                    drop_cols = [c for c in ["itemid", "bomc_id"] if c in extra_df.columns]
                    extra_df = extra_df.drop(columns=drop_cols, errors="ignore")
                if all(c in extra_df.columns for c in ["timestamp", "node_id", "kpi_name", "value"]):
                    all_dfs.append(extra_df[["timestamp", "node_id", "kpi_name", "value"]].copy())

        if not all_dfs:
            return pd.DataFrame(columns=["timestamp", "node_id", "kpi_name", "value"])

        combined = pd.concat(all_dfs, ignore_index=True)
        combined["value"] = pd.to_numeric(combined["value"], errors="coerce")
        combined["timestamp"] = pd.to_numeric(combined["timestamp"], errors="coerce")
        combined = combined.dropna(subset=["timestamp", "value"])
        return combined

    def load_trace(self, dataset: str, date_folder: str) -> pd.DataFrame:
        """Load trace_span.csv for a specific date."""
        path = os.path.join(
            self.dataset_dir, DATASET_DIRS[dataset],
            "telemetry", date_folder, "trace", "trace_span.csv",
        )
        if not os.path.exists(path):
            return pd.DataFrame()
        return pd.read_csv(path)

    def load_log(self, dataset: str, date_folder: str) -> pd.DataFrame:
        """Load log_service.csv for a specific date."""
        path = os.path.join(
            self.dataset_dir, DATASET_DIRS[dataset],
            "telemetry", date_folder, "log", "log_service.csv",
        )
        if not os.path.exists(path):
            return pd.DataFrame()
        return pd.read_csv(path)

    def build_metric_series_dict(
        self,
        dataset: str,
        date_folder: str,
        resample_interval: str = "60s",
    ) -> Dict[str, pd.DataFrame]:
        """Build a dict mapping node_id.kpi_name to resampled DataFrames.

        Each DataFrame has columns [timestamp, value].
        """
        combined = self.load_all_metrics(dataset, date_folder)
        if combined.empty:
            return {}

        result: Dict[str, pd.DataFrame] = {}
        for (node_id, kpi_name), group in combined.groupby(["node_id", "kpi_name"]):
            key = f"{node_id}.{kpi_name}"
            df = group[["timestamp", "value"]].copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
            df = df.dropna(subset=["timestamp"])
            df = df.sort_values("timestamp")
            df = df.set_index("timestamp")
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.resample(resample_interval).mean()
            df = df.reset_index()
            df["value"] = df["value"].interpolate(method="linear").ffill().bfill()
            result[key] = df[["timestamp", "value"]]

        return result

    def build_queries(self, dataset: str) -> List[RCAQuery]:
        """Parse all queries from query.csv into RCAQuery objects."""
        queries_df = self.load_queries(dataset)
        records_df = self.load_records(dataset)

        queries: List[RCAQuery] = []
        for idx, row in queries_df.iterrows():
            instruction = str(row.get("instruction", ""))
            scoring_points = str(row.get("scoring_points", ""))
            task_index = str(row.get("task_index", f"task_{idx}"))

            time_range = _parse_time_from_instruction(instruction)
            date_folder = _parse_date_folder(instruction)

            if time_range is None or time_range[0] is None:
                continue

            start_time, end_time = time_range
            start_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
            end_dt = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")

            start_ts = int(start_dt.timestamp())
            end_ts = int(end_dt.timestamp())

            task_type = _extract_task_type(instruction)

            target_fields = []
            if "component" in task_type:
                target_fields.append("component")
            if "reason" in task_type:
                target_fields.append("reason")
            if "time" in task_type:
                target_fields.append("time")
            if not target_fields:
                target_fields = ["component", "reason", "time"]

            candidate_components = self._extract_candidates_from_scoring(scoring_points, "component")
            candidate_reasons = self._extract_candidates_from_scoring(scoring_points, "reason")

            query = RCAQuery(
                dataset=dataset,
                row_id=int(idx),
                task_index=task_index,
                instruction=instruction,
                start_time=start_time,
                end_time=end_time,
                start_ts=start_ts,
                end_ts=end_ts,
                target_fields=target_fields,
                failure_count=1,
                candidate_components=candidate_components,
                candidate_reasons=candidate_reasons,
            )
            queries.append(query)

        return queries

    def get_ground_truth(
        self,
        dataset: str,
        start_ts: int,
        end_ts: int,
    ) -> List[Dict[str, Any]]:
        """Get ground truth records within a time window."""
        records_df = self.load_records(dataset)
        if records_df.empty:
            return []

        ts_col = "timestamp" if "timestamp" in records_df.columns else None
        if ts_col is None:
            return []

        records_df[ts_col] = pd.to_numeric(records_df[ts_col], errors="coerce")
        mask = (records_df[ts_col] >= start_ts) & (records_df[ts_col] <= end_ts)
        filtered = records_df[mask]
        return filtered.to_dict("records")

    def _extract_candidates_from_scoring(
        self,
        scoring_points: str,
        field: str,
    ) -> List[str]:
        """Extract candidate values from scoring_points text."""
        candidates = []
        if field == "component":
            import re
            m = re.search(r"root cause component is\s+(\S+)", scoring_points)
            if m:
                candidates.append(m.group(1).strip())
        elif field == "reason":
            import re
            m = re.search(r"root cause reason is\s+(.+?)(?:\n|$)", scoring_points)
            if m:
                candidates.append(m.group(1).strip())
        return candidates

    def list_date_folders(self, dataset: str) -> List[str]:
        """List available date folders for a dataset."""
        telemetry_dir = os.path.join(self.dataset_dir, DATASET_DIRS[dataset], "telemetry")
        if not os.path.isdir(telemetry_dir):
            return []
        folders = []
        for name in sorted(os.listdir(telemetry_dir)):
            if os.path.isdir(os.path.join(telemetry_dir, name)):
                folders.append(name)
        return folders
