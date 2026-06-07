"""OrchestratorAgent - CausalRCA-Flow 中央调度器。

技术方案 Agent 0 实现：
- OBSERVE-REASON-ACT-EVALUATE 循环（Agent Loop）
- 确定性调度规则（枚举决策）
- 质量门控 + 恢复预算机制
- 自适应阈值调整
- 级联workspace清理
"""
from __future__ import annotations

import logging
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional

from causalrca_codex.agents import (
    AssociationAgent,
    CausalGraphAgent,
    CounterfactualAgent,
    DataAgent,
    EvaluationAgent,
    FaultIdentificationAgent,
    InterventionAgent,
)
from causalrca_codex.config import AgentLoopConfig
from causalrca_codex.schemas import AgentResult, RCAQuery, RootCausePrediction
from causalrca_codex.workspace import make_workspace, snapshot_workspace

logger = logging.getLogger("causalrca")


# ---------------------------------------------------------------------------
# 打印工具
# ---------------------------------------------------------------------------
def _banner(msg: str, char: str = "=", width: int = 70) -> None:
    """打印醒目的分隔线标题。"""
    print(f"\n{char * width}")
    print(f"  {msg}")
    print(f"{char * width}")


def _step(iteration: int, agent: str, reason: str) -> None:
    """打印迭代步骤信息。"""
    _STEP_META = {
        "DataAgent": ("加载遥测数据+计算全天阈值", "metric.csv/trace.csv/log.csv"),
        "AssociationAgent": ("异常检测:阈值偏差+故障段", "data_layer.component_kpi_series"),
        "FaultIdentificationAgent": ("跨层粗过滤+主层选择+单组件短路", "association_layer.candidate_set"),
        "CausalGraphAgent": ("构建因果图(反转+Scheme-F)", "data_layer.raw_traces+fault_id_layer"),
        "InterventionAgent": ("干预评分(ES瓶颈+RCS多因子)", "causal_graph+fault_id_layer"),
        "CounterfactualAgent": ("反事实验证CES+原因识别", "intervention_layer.topk+raw_logs"),
    }
    purpose, data_src = _STEP_META.get(agent, ("-", "-"))
    print(f"\n  [{iteration:>2}] -> {agent:<30} ({reason})")
    print(f"        步骤目的: {purpose}")
    print(f"        读取数据: {data_src}")


def _quality_bar(label: str, quality: float) -> str:
    """生成可视化质量条。"""
    filled = int(quality * 20)
    bar = "#" * filled + "-" * (20 - filled)
    status = "OK" if quality >= 0.45 else "LOW"
    return f"  {label}: [{bar}] {quality:.2f} {status}"


class OrchestratorAgent:
    """质量门控的Agent Loop控制器（技术方案 Agent 0）。

    核心职责：
    1. 解析用户查询为结构化任务
    2. 运行 OBSERVE-REASON-ACT-EVALUATE 循环
    3. 根据质量评估触发恢复策略
    4. 最终整合所有Agent输出为RootCausePrediction

    调度规则（确定性枚举，非LLM推理）：
    - 数据未加载 -> DataAgent
    - 异常未检测 -> AssociationAgent
    - 故障识别未完成 -> FaultIdentificationAgent
    - 单组件短路 -> 跳过因果图 -> InterventionAgent
    - 多组件 -> CausalGraphAgent -> InterventionAgent
    - 反事实验证 -> CounterfactualAgent -> FINALIZE
    """

    def __init__(self, config: Optional[AgentLoopConfig] = None) -> None:
        self.config = config or AgentLoopConfig()
        self.agents = {
            "DataAgent": DataAgent(self.config),
            "AssociationAgent": AssociationAgent(self.config),
            "FaultIdentificationAgent": FaultIdentificationAgent(self.config),
            "CausalGraphAgent": CausalGraphAgent(self.config),
            "InterventionAgent": InterventionAgent(self.config),
            "CounterfactualAgent": CounterfactualAgent(self.config),
            "EvaluationAgent": EvaluationAgent(self.config),
        }
        self.workspace = make_workspace()
        self.recovery_budget = deepcopy(self.config.recovery_budget)
        self.params = {
            "DataAgent": {
                "threshold_percentile": self.config.threshold_percentile,
                "low_percentile": self.config.low_percentile,
            },
            "AssociationAgent": {
                "min_fault_points": self.config.min_fault_points,
                "beta_min": self.config.beta_min,
                "severity_threshold": self.config.severity_threshold,
            },
            "FaultIdentificationAgent": {"tau_single": self.config.tau_single},
            "CausalGraphAgent": {"expand_mode": self.config.expand_mode},
            "InterventionAgent": {},
            "CounterfactualAgent": {},
        }

    def run(self, query: RCAQuery, ground_truth: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """主入口：运行完整的RCA流程。

        Args:
            query: 解析后的RCA查询对象
            ground_truth: 可选的基准答案（用于评估模式）

        Returns:
            dict: 包含prediction、prediction_json、workspace、trajectory、diagnostics
        """
        self.workspace = make_workspace()
        self.recovery_budget = deepcopy(self.config.recovery_budget)
        self.workspace["task"] = {
            "query": query,
            "system": query.dataset,
            "time_start": query.start_time,
            "time_end": query.end_time,
            "required_elements": list(query.target_fields),
        }
        if ground_truth:
            self.workspace["ground_truth"] = ground_truth

        _banner(f"CausalRCA-Flow | {query.dataset} | Row {query.row_id}")
        print(f"  查询: {query.instruction}")
        print(f"  时间窗口: {query.start_time} ~ {query.end_time}")
        print(f"  候选组件: {len(query.candidate_components)} | 候选原因: {len(query.candidate_reasons)}")
        print(f"  最大迭代: {self.config.max_iterations} | LLM: {'ON' if self.config.use_llm_reasoning else 'OFF'}")

        start_time = time.time()

        for iteration in range(self.config.max_iterations):
            state = self.observe()
            plan = self.reason_next_action(state)

            if plan["agent"] == "FINALIZE":
                print(f"\n  [{'FIN':>3}] 所有层完成，准备输出最终结果")
                break

            _step(iteration, plan["agent"], plan["reason"])

            action_result = self.act(plan)
            evaluation = self.evaluate(action_result, plan)
            self.update_workspace_history(iteration, plan, action_result, evaluation)

            # 打印质量评估
            print(_quality_bar("自评", action_result.self_assessed_quality))
            print(_quality_bar("全局", evaluation["global_quality"]))

            if evaluation["status"] in {"FAILED", "LOW_QUALITY"}:
                print(f"  [!] 状态={evaluation['status']}，触发恢复策略...")
                recovery = self.plan_recovery(state, plan, action_result, evaluation)
                if recovery["action"] == "retry":
                    self.workspace["recovery_history"].append(recovery)
                    print(f"  [R] 恢复: {recovery['reason']} (剩余预算={recovery.get('remaining_budget', '?')})")
                    continue
                if recovery["action"] == "accept_low_confidence":
                    self.workspace["recovery_history"].append(recovery)
                    print(f"  [>>] 接受低置信度: {recovery.get('reason', '预算耗尽')}")

            if self.is_task_complete():
                print(f"\n  [OK] 任务完成（final_root_cause已设置）")
                break

        # 评估模式
        if ground_truth:
            eval_result = self.agents["EvaluationAgent"].run(self.workspace, {"ground_truth": ground_truth})
            self.workspace["history"].append(
                {"agent": "EvaluationAgent", "result": eval_result.to_dict(), "global_quality": 1.0}
            )

        elapsed = time.time() - start_time
        result = self.finalize()

        # 打印最终结果
        pred = result.get("prediction")
        if pred:
            _banner("最终诊断结果", char="-")
            print(f"  根因组件: {pred.component}")
            print(f"  故障时间: {pred.occurrence_time}")
            print(f"  故障原因: {pred.reason}")
            print(f"  最终得分: {pred.scores.get('FinalScore', 0):.4f}")
            print(f"  说明: {pred.explanation}")
        print(f"\n  总耗时: {elapsed:.2f}s | 迭代次数: {iteration + 1}")

        return result

    def observe(self) -> Dict[str, Any]:
        """OBSERVE: 从workspace提取当前状态摘要。"""
        ws = self.workspace
        graph = ws.get("causal_graph_layer", {}).get("weighted_causal_graph")
        return {
            "task": ws.get("task", {}),
            "current_stage": ws.get("current_stage"),
            "data_quality": ws.get("data_layer", {}).get("data_quality", {}),
            "anomaly_count": len(ws.get("association_layer", {}).get("candidate_set", [])),
            "refined_count": len(ws.get("fault_id_layer", {}).get("refined_candidates", [])),
            "needs_causal_inference": ws.get("fault_id_layer", {}).get("needs_causal_inference"),
            "graph_nodes": len(graph.nodes) if graph is not None else 0,
            "graph_edges": len(graph.edges) if graph is not None else 0,
            "top1_confidence": ws.get("intervention_layer", {}).get("top1_confidence", 0.0),
            "counterfactual_quality": ws.get("counterfactual_layer", {}).get("quality", 0.0),
            "last_warnings": self.collect_recent_warnings(k=3),
            "history_tail": ws.get("history", [])[-3:],
        }

    def reason_next_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """REASON: 确定性调度决策（枚举规则，非LLM推理）。

        技术方案要求：decision rules are enumerated。
        调度逻辑按优先级检查workspace各层的完成状态。
        """
        ws = self.workspace

        # 规则1：数据层为空 -> DataAgent
        if not ws["data_layer"].get("component_kpi_series"):
            return {"agent": "DataAgent", "params": self.params["DataAgent"], "reason": "数据层未加载"}

        # 规则2：关联层为空 -> AssociationAgent
        if not ws["association_layer"].get("candidate_set"):
            return {"agent": "AssociationAgent", "params": self.params["AssociationAgent"], "reason": "异常检测未完成"}

        # 规则3：故障识别层为空 -> FaultIdentificationAgent
        if ws["fault_id_layer"].get("needs_causal_inference") is None:
            return {
                "agent": "FaultIdentificationAgent",
                "params": self.params["FaultIdentificationAgent"],
                "reason": "故障识别未完成",
            }

        # 规则4：单组件短路 -> 跳过因果图
        if ws["fault_id_layer"].get("needs_causal_inference") is False and self.config.allow_single_component_shortcut:
            if not ws["intervention_layer"].get("ranking"):
                return {"agent": "InterventionAgent", "params": self.params["InterventionAgent"], "reason": "单组件短路-评分"}
            if not ws["counterfactual_layer"].get("final_root_cause"):
                return {
                    "agent": "CounterfactualAgent",
                    "params": self.params["CounterfactualAgent"],
                    "reason": "单组件短路-验证",
                }
            return {"agent": "FINALIZE", "reason": "单组件路径完成"}

        # 规则5：多组件 -> 因果图构建
        if ws["causal_graph_layer"].get("weighted_causal_graph") is None:
            return {"agent": "CausalGraphAgent", "params": self.params["CausalGraphAgent"], "reason": "因果图未构建"}

        # 规则6：干预排名
        if not ws["intervention_layer"].get("ranking"):
            return {"agent": "InterventionAgent", "params": self.params["InterventionAgent"], "reason": "干预排名未完成"}

        # 规则7：反事实验证
        if not ws["counterfactual_layer"].get("final_root_cause"):
            return {
                "agent": "CounterfactualAgent",
                "params": self.params["CounterfactualAgent"],
                "reason": "反事实验证未完成",
            }

        return {"agent": "FINALIZE", "reason": "所有必需层已完成"}

    def act(self, plan: Dict[str, Any]) -> AgentResult:
        """ACT: 分派到指定Agent执行。"""
        agent_name = plan["agent"]
        self.workspace["current_stage"] = agent_name
        agent = self.agents[agent_name]
        import time as _time
        _t0 = _time.time()
        result = agent.run(self.workspace, plan.get("params") or {})
        _elapsed = _time.time() - _t0
        # Store per-agent timing in workspace for progress reporting
        if "_agent_timings" not in self.workspace:
            self.workspace["_agent_timings"] = {}
        self.workspace["_agent_timings"][agent_name] = self.workspace["_agent_timings"].get(agent_name, 0.0) + _elapsed
        print(f"    ⏱ {agent_name}: {_elapsed:.1f}s")
        return result

    def evaluate(self, action_result: AgentResult, plan: Dict[str, Any]) -> Dict[str, Any]:
        """EVALUATE: 混合自评估和外部验证的质量评分。

        公式：global_quality = alpha * self_quality + (1-alpha) * validation
        """
        validation = self.external_validation_score(action_result.agent_name, action_result.result)
        global_quality = (
            self.config.quality_alpha * action_result.self_assessed_quality
            + (1.0 - self.config.quality_alpha) * validation
        )
        layer = self._layer_for_agent(action_result.agent_name)
        if layer and layer in self.workspace:
            self.workspace[layer]["quality"] = round(global_quality, 6)
            self.workspace[layer]["warnings"] = action_result.warnings
        status = "OK"
        if action_result.status != "OK":
            status = "FAILED"
        elif global_quality < self.config.low_quality_threshold:
            status = "LOW_QUALITY"
        return {
            "status": status,
            "agent_quality": action_result.self_assessed_quality,
            "validation_quality": validation,
            "global_quality": round(global_quality, 6),
        }

    def external_validation_score(self, agent_name: str, result: Dict[str, Any]) -> float:
        """外部验证分数：基于客观指标的独立质量评估。"""
        if agent_name == "DataAgent":
            quality = result.get("data_quality", {})
            return 1.0 if quality.get("metric_series", 0) and quality.get("series_with_window_rows", 0) else 0.2
        if agent_name == "AssociationAgent":
            count = len(result.get("candidate_set", []))
            if count == 0:
                return 0.1
            if count <= 10:
                return 0.95
            if count <= 30:
                return 0.70
            return 0.35
        if agent_name == "FaultIdentificationAgent":
            return 0.9 if result.get("refined_candidates") else 0.1
        if agent_name == "CausalGraphAgent":
            if result.get("edges"):
                return 0.85
            return 0.45 if result.get("nodes") else 0.1
        if agent_name == "InterventionAgent":
            return min(1.0, max(0.1, float(result.get("top1_confidence", 0.0))))
        if agent_name == "CounterfactualAgent":
            final = result.get("final_root_cause", {})
            scores = final.get("scores", {})
            return min(1.0, 0.4 + float(scores.get("ReasonScore", 0.0)))
        return 1.0

    def plan_recovery(
        self,
        state: Dict[str, Any],
        plan: Dict[str, Any],
        action_result: AgentResult,
        evaluation: Dict[str, Any],
    ) -> Dict[str, Any]:
        """恢复策略：针对每个Agent的特定恢复方案（带预算限制）。

        技术方案恢复规则：
        - AssociationAgent: 0候选->降低阈值, >30->提高阈值
        - FaultIdentificationAgent: 恢复reserve候选
        - CausalGraphAgent: expand_mode升级
        - InterventionAgent: 扩展图重建
        - CounterfactualAgent: 扩大Top-K
        """
        agent_name = action_result.agent_name
        budget = self.recovery_budget.get(agent_name, 0)
        if budget <= 0:
            return {
                "agent": agent_name,
                "action": "accept_low_confidence",
                "reason": "恢复预算耗尽",
                "evaluation": evaluation,
            }
        self.recovery_budget[agent_name] = budget - 1

        if agent_name == "DataAgent":
            self.params["DataAgent"]["aggregate"] = "max"
            self._clear_from("data")
            return self._retry(agent_name, "使用max聚合重试数据加载", evaluation)

        if agent_name == "AssociationAgent":
            count = len(action_result.result.get("candidate_set", []))
            if count == 0:
                # 0候选: 先尝试扩展窗口（RootCandidateRecovery），再降低阈值
                current_expand = int(self.params["DataAgent"].get("window_expand_minutes", 0))
                if current_expand < 10:
                    self.params["DataAgent"]["window_expand_minutes"] = 10
                    self._clear_from("data")
                    return self._retry(agent_name, "扩展metric窗口-10min (RootCandidateRecovery)", evaluation)
                current = float(self.params["DataAgent"].get("threshold_percentile", self.config.threshold_percentile))
                self.params["DataAgent"]["threshold_percentile"] = 90.0 if current >= 95.0 else 85.0
                self.params["AssociationAgent"]["beta_min"] = max(0.25, float(self.params["AssociationAgent"]["beta_min"]) - 0.15)
                self._clear_from("data")
                return self._retry(agent_name, "降低阈值百分位并重跑数据/关联", evaluation)
            # >30候选：提高阈值
            self.params["DataAgent"]["threshold_percentile"] = 99.0
            self.params["AssociationAgent"]["min_fault_points"] = int(self.params["AssociationAgent"]["min_fault_points"]) + 1
            self._clear_from("data")
            return self._retry(agent_name, "提高阈值并要求更长故障段", evaluation)

        if agent_name == "FaultIdentificationAgent":
            self.params["FaultIdentificationAgent"]["restore_reserve"] = True
            self.params["FaultIdentificationAgent"]["force_multi_component"] = True
            self._clear_from("fault")
            return self._retry(agent_name, "恢复reserve候选池并强制多组件推理", evaluation)

        if agent_name == "CausalGraphAgent":
            mode = self.params["CausalGraphAgent"].get("expand_mode", "none")
            self.params["CausalGraphAgent"]["expand_mode"] = "direct" if mode == "none" else "full_path"
            self._clear_from("graph")
            return self._retry(agent_name, f"扩展因果图 ({mode} -> {self.params['CausalGraphAgent']['expand_mode']})", evaluation)

        if agent_name == "InterventionAgent":
            mode = self.params["CausalGraphAgent"].get("expand_mode", "none")
            self.params["CausalGraphAgent"]["expand_mode"] = "direct" if mode == "none" else "full_path"
            self._clear_from("graph")
            return self._retry(agent_name, "干预置信度低，重建扩展图", evaluation)

        if agent_name == "CounterfactualAgent":
            self.params["FaultIdentificationAgent"]["force_multi_component"] = True
            self.params["FaultIdentificationAgent"]["restore_reserve"] = True
            self._clear_from("fault")
            return self._retry(agent_name, "原因证据不足，扩大Top-K重新验证", evaluation)

        return {"agent": agent_name, "action": "accept_low_confidence", "evaluation": evaluation}

    def is_task_complete(self) -> bool:
        """检查任务是否已完成（final_root_cause已设置）。"""
        final = self.workspace.get("final_root_cause")
        self.workspace["task_complete"] = final is not None
        return self.workspace["task_complete"]

    def finalize(self) -> Dict[str, Any]:
        """最终化：打包预测结果、workspace快照、轨迹和诊断信息。"""
        query: RCAQuery = self.workspace["task"]["query"]
        final = self.workspace.get("final_root_cause")
        if final is None:
            final = self._fallback_prediction(query)
            self.workspace["final_root_cause"] = final
        return {
            "prediction": final,
            "prediction_json": final.to_opencra_json(query.target_fields),
            "workspace": snapshot_workspace(self.workspace),
            "trajectory": list(self.workspace["history"]),
            "diagnostics": {
                "recovery_history": list(self.workspace["recovery_history"]),
                "evaluation": self.workspace.get("evaluation_layer", {}),
            },
        }

    def update_workspace_history(
        self,
        iteration: int,
        plan: Dict[str, Any],
        action_result: AgentResult,
        evaluation: Dict[str, Any],
    ) -> None:
        """记录迭代历史到workspace。"""
        self.workspace["history"].append(
            {
                "iteration": iteration,
                "plan": plan,
                "agent": action_result.agent_name,
                "result": action_result.to_dict(),
                "evaluation": evaluation,
            }
        )

    def collect_recent_warnings(self, k: int = 3) -> List[str]:
        """收集最近k次迭代的警告信息。"""
        warnings: List[str] = []
        for item in self.workspace.get("history", [])[-k:]:
            warnings.extend(item.get("result", {}).get("warnings", []))
        return warnings

    def _layer_for_agent(self, agent_name: str) -> Optional[str]:
        """Agent名称到workspace层名的映射。"""
        return {
            "DataAgent": "data_layer",
            "AssociationAgent": "association_layer",
            "FaultIdentificationAgent": "fault_id_layer",
            "CausalGraphAgent": "causal_graph_layer",
            "InterventionAgent": "intervention_layer",
            "CounterfactualAgent": "counterfactual_layer",
            "EvaluationAgent": "evaluation_layer",
        }.get(agent_name)

    def _retry(self, agent_name: str, reason: str, evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """构造重试恢复响应。"""
        return {
            "agent": agent_name,
            "action": "retry",
            "reason": reason,
            "remaining_budget": self.recovery_budget.get(agent_name, 0),
            "params": deepcopy(self.params),
            "evaluation": evaluation,
        }

    def _clear_from(self, stage: str) -> None:
        """级联清理workspace：清除指定阶段及其所有下游层。

        清理规则：
        - data -> association -> fault -> graph -> intervention -> counterfactual
        - graph -> intervention -> counterfactual
        """
        if stage == "data":
            self.workspace["data_layer"].update(
                {
                    "component_kpi_series": {},
                    "full_day_series": {},
                    "global_thresholds": {},
                    "raw_metrics": {},
                    "raw_traces": [],
                    "raw_logs": [],
                }
            )
            self._clear_from("association")
        elif stage == "association":
            self.workspace["association_layer"].update(
                {
                    "candidate_set": [],
                    "anomaly_details": {},
                    "anomaly_scores": {},
                    "component_intensity_series": {},
                }
            )
            self.workspace["fault_id_layer"]["reserve_candidates"] = []
            self._clear_from("fault")
        elif stage == "fault":
            self.workspace["fault_id_layer"].update(
                {
                    "refined_candidates": [],
                    "primary_layer": None,
                    "needs_causal_inference": None,
                    "tentative_root_cause": None,
                }
            )
            self._clear_from("graph")
        elif stage == "graph":
            self.workspace["causal_graph_layer"].update(
                {
                    "local_call_graph": {},
                    "weighted_causal_graph": None,
                    "edge_scores": {},
                    "quality": 0.0,
                }
            )
            self.workspace["intervention_layer"].update(
                {
                    "explain_scores": {},
                    "root_cause_scores": {},
                    "ranking": [],
                    "top1_confidence": 0.0,
                    "topk_candidates": [],
                }
            )
            self.workspace["counterfactual_layer"].update(
                {
                    "contextual_explain_scores": {},
                    "reason_scores": {},
                    "final_root_cause": None,
                }
            )
            self.workspace["final_root_cause"] = None

    def _fallback_prediction(self, query: RCAQuery) -> RootCausePrediction:
        """降级预测：当Agent循环未正常终结时，选择最高分异常组件作为兜底。"""
        scores = self.workspace["association_layer"].get("anomaly_scores", {})
        component = max(scores.items(), key=lambda item: item[1])[0] if scores else (query.candidate_components[0] if query.candidate_components else "")
        details = self.workspace["association_layer"].get("anomaly_details", {}).get(component, [])
        reason = ""
        if details:
            reason = details[0].get("reason_hint") or ""
        if not reason and query.candidate_reasons:
            reason = query.candidate_reasons[0]
        time = details[0].get("start_time") if details else query.start_time
        return RootCausePrediction(
            component=component,
            occurrence_time=time,
            reason=reason,
            scores={"FinalScore": float(scores.get(component, 0.0))},
            explanation="降级预测：Agent循环未正常终结，选择最高异常分数组件作为兜底。",
        )
