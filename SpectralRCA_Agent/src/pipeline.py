from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.config import SpectralRCAConfig
from src.coordinator import Coordinator
from src.data_loader import DataLoader
from src.schemas import RCAQuery, RCAResult


class SpectralRCAPipeline:
    """Full pipeline for SpectralRCA-Agent evaluation.

    Supports:
    - Running the full pipeline on all queries in a dataset
    - Running ablation experiments (disabling specific innovations)
    - Computing evaluation metrics (component accuracy, reason accuracy, time accuracy)
    - Saving results to disk
    """

    def __init__(
        self,
        dataset_dir: str,
        config: Optional[SpectralRCAConfig] = None,
        output_dir: Optional[str] = None,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.config = config or SpectralRCAConfig()
        self.config.dataset_dir = dataset_dir
        self.output_dir = output_dir or os.path.join(dataset_dir, "results")
        self.data_loader = DataLoader(dataset_dir)
        self.coordinator = Coordinator(self.config, self.data_loader)

    def run_dataset(
        self,
        dataset: str,
        max_queries: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run the pipeline on all queries for a dataset.

        Args:
            dataset: Dataset name ('Bank' or 'Telecom').
            max_queries: Maximum number of queries to process.

        Returns:
            Evaluation results dict.
        """
        queries = self.data_loader.build_queries(dataset)
        if max_queries:
            queries = queries[:max_queries]

        results: List[Dict[str, Any]] = []
        for i, query in enumerate(queries):
            print(f"[{i+1}/{len(queries)}] Processing {query.task_index}...")
            try:
                rca_result = self.coordinator.run(query)
                result = self._evaluate_single(query, rca_result)
                results.append(result)
            except Exception as e:
                print(f"  Error: {e}")
                results.append({
                    "task_index": query.task_index,
                    "error": str(e),
                })

        evaluation = self._compute_overall_metrics(results)
        evaluation["per_query_results"] = results

        self._save_results(evaluation, dataset)
        return evaluation

    def run_single(
        self,
        dataset: str,
        date_folder: str,
        incident_start: str,
        incident_end: str,
        instruction: str = "",
    ) -> RCAResult:
        """Run the pipeline on a single incident.

        Args:
            dataset: Dataset name.
            date_folder: Date folder name (e.g., '2021_03_04').
            incident_start: Incident start time string.
            incident_end: Incident end time string.
            instruction: Query instruction text.

        Returns:
            RCAResult for this incident.
        """
        query = RCAQuery(
            dataset=dataset,
            row_id=0,
            task_index=f"single_{date_folder}",
            instruction=instruction or f"On {date_folder}, within the time range of {incident_start} to {incident_end}",
            start_time=incident_start,
            end_time=incident_end,
            start_ts=int(datetime.strptime(incident_start, "%Y-%m-%d %H:%M:%S").timestamp()),
            end_ts=int(datetime.strptime(incident_end, "%Y-%m-%d %H:%M:%S").timestamp()),
            target_fields=["component", "reason", "time"],
            failure_count=1,
            candidate_components=[],
            candidate_reasons=[],
        )

        return self.coordinator.run(query)

    def run_ablation(
        self,
        dataset: str,
        ablation_config: str,
        max_queries: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run ablation experiment with specific features disabled.

        Args:
            dataset: Dataset name.
            ablation_config: One of:
                - 'no_spectral': Disable spectral anomaly detection
                - 'no_abductive': Disable abductive reasoning
                - 'no_memory': Disable self-evolving memory
                - 'no_graph_spectral': Disable spectral edge validation
                - 'traditional_only': Use only traditional anomaly detection
            max_queries: Maximum number of queries.

        Returns:
            Ablation evaluation results.
        """
        config = SpectralRCAConfig()

        if ablation_config == "no_spectral":
            config.enable_spectral_anomaly = False
        elif ablation_config == "no_abductive":
            config.enable_abductive = False
        elif ablation_config == "no_memory":
            config.enable_memory = False
        elif ablation_config == "no_graph_spectral":
            config.enable_spectral_shape = False
            config.enable_dominant_freq_match = False
            config.enable_phase_lag = False
        elif ablation_config == "traditional_only":
            config.enable_spectral_anomaly = False
            config.enable_abductive = False
            config.enable_memory = False
            config.enable_spectral_shape = False
            config.enable_dominant_freq_match = False
            config.enable_phase_lag = False
        else:
            raise ValueError(f"Unknown ablation config: {ablation_config}")

        coordinator = Coordinator(config, self.data_loader)

        queries = self.data_loader.build_queries(dataset)
        if max_queries:
            queries = queries[:max_queries]

        results: List[Dict[str, Any]] = []
        for i, query in enumerate(queries):
            try:
                rca_result = coordinator.run(query)
                result = self._evaluate_single(query, rca_result)
                results.append(result)
            except Exception as e:
                results.append({"task_index": query.task_index, "error": str(e)})

        evaluation = self._compute_overall_metrics(results)
        evaluation["ablation_config"] = ablation_config
        evaluation["per_query_results"] = results
        return evaluation

    def _evaluate_single(self, query: RCAQuery, result: RCAResult) -> Dict[str, Any]:
        """Evaluate a single query result against ground truth."""
        ground_truth = self.data_loader.get_ground_truth(
            query.dataset, query.start_ts, query.end_ts,
        )

        predicted = json.loads(result.prediction_json) if result.prediction_json else {}

        gt_component = None
        gt_reason = None
        gt_time = None

        if ground_truth:
            gt = ground_truth[0]
            gt_component = gt.get("component", gt.get("cmdb_id"))
            gt_reason = gt.get("reason")
            gt_ts = gt.get("timestamp")
            if gt_ts:
                try:
                    gt_time = datetime.fromtimestamp(float(gt_ts)).strftime("%Y-%m-%d %H:%M:%S")
                except (ValueError, OSError):
                    gt_time = None

        pred_component = predicted.get("component")
        pred_reason = predicted.get("reason")
        pred_time = predicted.get("time")

        component_match = False
        if gt_component and pred_component:
            component_match = gt_component.lower().strip() == pred_component.lower().strip()

        reason_match = False
        if gt_reason and pred_reason:
            gt_reason_lower = gt_reason.lower().strip()
            pred_reason_lower = pred_reason.lower().strip()
            reason_match = (
                gt_reason_lower == pred_reason_lower
                or gt_reason_lower in pred_reason_lower
                or pred_reason_lower in gt_reason_lower
            )

        time_match = False
        if gt_time and pred_time and pred_time != "unknown":
            try:
                gt_dt = datetime.strptime(gt_time, "%Y-%m-%d %H:%M:%S")
                pred_dt = datetime.strptime(pred_time, "%Y-%m-%d %H:%M:%S")
                time_match = abs((pred_dt - gt_dt).total_seconds()) <= 60
            except ValueError:
                time_match = False

        top5_match = False
        if gt_component:
            for cand in result.ranked_candidates[:5]:
                if cand.get("node_id", "").lower().strip() == gt_component.lower().strip():
                    top5_match = True
                    break

        return {
            "task_index": query.task_index,
            "gt_component": gt_component,
            "pred_component": pred_component,
            "component_match": component_match,
            "gt_reason": gt_reason,
            "pred_reason": pred_reason,
            "reason_match": reason_match,
            "gt_time": gt_time,
            "pred_time": pred_time,
            "time_match": time_match,
            "top5_match": top5_match,
            "top_score": result.ranked_candidates[0].get("root_score", 0.0) if result.ranked_candidates else 0.0,
        }

    def _compute_overall_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute overall evaluation metrics."""
        valid = [r for r in results if "error" not in r]
        if not valid:
            return {"total": 0, "valid": 0}

        total = len(valid)
        component_correct = sum(1 for r in valid if r.get("component_match", False))
        reason_correct = sum(1 for r in valid if r.get("reason_match", False))
        time_correct = sum(1 for r in valid if r.get("time_match", False))
        top5_correct = sum(1 for r in valid if r.get("top5_match", False))

        full_match = sum(
            1 for r in valid
            if r.get("component_match", False)
            and r.get("reason_match", False)
            and r.get("time_match", False)
        )

        component_tasks = [r for r in valid if "component" in r.get("gt_component", "")]
        reason_tasks = [r for r in valid if r.get("gt_reason")]

        return {
            "total_queries": len(results),
            "valid_queries": total,
            "component_accuracy": round(component_correct / max(total, 1), 4),
            "reason_accuracy": round(reason_correct / max(total, 1), 4),
            "time_accuracy": round(time_correct / max(total, 1), 4),
            "top5_accuracy": round(top5_correct / max(total, 1), 4),
            "full_match_rate": round(full_match / max(total, 1), 4),
            "component_correct": component_correct,
            "reason_correct": reason_correct,
            "time_correct": time_correct,
            "top5_correct": top5_correct,
            "full_match_correct": full_match,
        }

    def _save_results(self, results: Dict[str, Any], dataset: str) -> None:
        """Save evaluation results to disk."""
        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.output_dir, f"{dataset}_results_{timestamp}.json")

        serializable = self._make_serializable(results)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)

        print(f"Results saved to {path}")

    def _make_serializable(self, obj: Any) -> Any:
        """Make an object JSON serializable."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        elif isinstance(obj, tuple):
            return list(obj)
        else:
            return str(obj)
