from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict


def make_workspace() -> Dict[str, Any]:
    return {
        "task": {},
        "data_layer": {
            "component_kpi_series": {},
            "full_day_series": {},
            "global_thresholds": {},
            "raw_metrics": {},
            "raw_traces": [],
            "raw_logs": [],
            "data_quality": {},
            "params_used": {},
        },
        "association_layer": {
            "candidate_set": [],
            "anomaly_details": {},
            "anomaly_scores": {},
            "component_intensity_series": {},
            "params_used": {},
            "quality": 0.0,
            "warnings": [],
        },
        "fault_id_layer": {
            "refined_candidates": [],
            "reserve_candidates": [],
            "primary_layer": None,
            "needs_causal_inference": None,
            "tentative_root_cause": None,
            "quality": 0.0,
            "warnings": [],
        },
        "causal_graph_layer": {
            "local_call_graph": {},
            "local_causal_graph": None,
            "weighted_causal_graph": None,
            "edge_scores": {},
            "graph_config": {},
            "quality": 0.0,
            "warnings": [],
        },
        "intervention_layer": {
            "explain_scores": {},
            "root_cause_scores": {},
            "ranking": [],
            "top1_confidence": 0.0,
            "topk_candidates": [],
            "warnings": [],
        },
        "counterfactual_layer": {
            "contextual_explain_scores": {},
            "reason_scores": {},
            "final_root_cause": None,
            "counterfactual_explanation": "",
            "quality": 0.0,
            "warnings": [],
        },
        "evaluation_layer": {},
        "current_stage": None,
        "history": [],
        "recovery_history": [],
        "final_root_cause": None,
        "task_complete": False,
    }


def snapshot_workspace(workspace: Dict[str, Any]) -> Dict[str, Any]:
    copied = deepcopy(workspace)
    graph = copied.get("causal_graph_layer", {}).get("weighted_causal_graph")
    if hasattr(graph, "to_dict"):
        copied["causal_graph_layer"]["weighted_causal_graph"] = graph.to_dict()
    return copied
