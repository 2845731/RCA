from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Tuple

from causalrca_codex.agents.base import BaseAgent
from causalrca_codex.schemas import RootCausePrediction


class EvaluationAgent(BaseAgent):
    """Agent 7: 评估Agent - 逐步失败归因（技术方案 Step 7, 仅评估模式）。

    职责：给定基准答案，诊断Pipeline哪个阶段首先失败。
    六种失败类型（Case A-F）：
    - Case A (AD-FN): 真实根因不在异常候选集中
    - Case A (AD-FP-NOISE): 候选过多(>30)
    - Case B (FI-MISFILTER): 真实根因在候选中但被层过滤移除
    - Case C (INT-RANK): 单组件场景识别错误
    - Case D (INT-RANK): 多组件场景Top-1不是真实根因
    - Case E (CF-REASON): 组件正确但原因错误
    - Case F (CF-TIME): 组件和原因正确但时间错误
    """

    name = "EvaluationAgent"
    purpose = "逐步失败归因：定位Pipeline哪个阶段导致诊断错误"
    preconditions = ["workspace.final_root_cause", "ground_truth"]
    produces = ["evaluation_layer.failure_type", "evaluation_layer.first_failed_stage"]

    def _execute(self, workspace: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        ground_truth = params.get("ground_truth") or workspace.get("ground_truth")
        if not ground_truth:
            return {"evaluation_skipped": True, "reason": "ground_truth not provided"}

        prediction = workspace.get("final_root_cause")
        query = workspace["task"]["query"]
        target_fields = set(query.target_fields)
        predicted_component = prediction.component if prediction else ""
        predicted_reason = prediction.reason if prediction else ""
        predicted_time = prediction.occurrence_time if prediction else ""
        gt_component = str(ground_truth.get("component", ""))
        gt_reason = str(ground_truth.get("reason", ""))
        gt_time = str(ground_truth.get("datetime", ""))

        component_required = "root cause component" in target_fields
        reason_required = "root cause reason" in target_fields
        time_required = "root cause occurrence datetime" in target_fields
        final_correct = True
        if component_required and gt_component:
            final_correct = final_correct and predicted_component == gt_component
        if reason_required and gt_reason:
            final_correct = final_correct and predicted_reason == gt_reason
        if time_required and gt_time:
            final_correct = final_correct and self._time_close(predicted_time, gt_time)
        failure_type = None
        first_stage = None
        evidence: Dict[str, Any] = {"gt_component": gt_component, "gt_reason": gt_reason, "gt_time": gt_time}

        candidate_set = workspace["association_layer"].get("candidate_set", [])
        refined = workspace["fault_id_layer"].get("refined_candidates", [])
        graph = workspace["causal_graph_layer"].get("weighted_causal_graph")
        ranking = workspace["intervention_layer"].get("ranking", [])

        if gt_component and gt_component not in candidate_set:
            failure_type = "AD-FN"
            first_stage = "AssociationAgent"
            evidence["candidate_set"] = candidate_set
        elif len(candidate_set) > 30:
            failure_type = "AD-FP-NOISE"
            first_stage = "AssociationAgent"
            evidence["candidate_set_size"] = len(candidate_set)
        elif gt_component and gt_component in candidate_set and gt_component not in refined:
            failure_type = "FI-MISFILTER"
            first_stage = "FaultIdentificationAgent"
            evidence["refined_candidates"] = refined
        elif gt_component and graph is not None and not graph.has_node(gt_component):
            failure_type = "CG-MISSING-NODE"
            first_stage = "CausalGraphAgent"
        elif component_required and gt_component and ranking and ranking[0].get("component") != gt_component:
            gt_rank = next((idx + 1 for idx, row in enumerate(ranking) if row.get("component") == gt_component), None)
            failure_type = "INT-RANK"
            first_stage = "InterventionAgent"
            evidence["gt_rank"] = gt_rank
            evidence["top1"] = ranking[0].get("component")
        elif reason_required and gt_component and predicted_component == gt_component and gt_reason and predicted_reason != gt_reason:
            failure_type = "CF-REASON"
            first_stage = "CounterfactualAgent"
            evidence["predicted_reason"] = predicted_reason
        elif time_required and gt_component and predicted_component == gt_component and gt_time and not self._time_close(predicted_time, gt_time):
            failure_type = "CF-TIME"
            first_stage = "CounterfactualAgent"
            evidence["predicted_time"] = predicted_time
        elif not final_correct:
            failure_type = "TOOL-ERROR"
            first_stage = "Unknown"

        report = {
            "case_id": f"{workspace['task']['query'].dataset}_{workspace['task']['query'].row_id}",
            "final_correct": final_correct,
            "first_failed_stage": first_stage,
            "failure_type": failure_type,
            "evidence": evidence,
        }
        workspace["evaluation_layer"] = report

        # 醒目打印评估结果
        status = "[OK] CORRECT" if final_correct else "[X] WRONG"
        print(f"    [Evaluation] {status}")
        if failure_type:
            print(f"    [Evaluation] 失败类型: {failure_type} | 首个失败阶段: {first_stage}")
            if "gt_rank" in evidence:
                print(f"    [Evaluation] 真实根因排名: #{evidence['gt_rank']} Top1: {evidence.get('top1', '?')}")

        return report

    def _time_close(self, predicted: str, expected: str, tolerance_seconds: int = 60) -> bool:
        if not predicted or not expected:
            return True
        fmt = "%Y-%m-%d %H:%M:%S"
        try:
            return abs((datetime.strptime(predicted, fmt) - datetime.strptime(expected, fmt)).total_seconds()) <= tolerance_seconds
        except ValueError:
            return False

    def _self_evaluate(
        self,
        result: Dict[str, Any],
        workspace: Dict[str, Any],
        params: Dict[str, Any],
    ) -> Tuple[float, List[str]]:
        if result.get("evaluation_skipped"):
            return 1.0, []
        warnings = [] if result.get("final_correct") else ["Prediction failed; step-wise attribution was generated."]
        return 1.0, warnings
