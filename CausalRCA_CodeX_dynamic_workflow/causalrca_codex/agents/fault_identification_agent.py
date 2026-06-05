from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Tuple

from causalrca_codex.agents.base import BaseAgent
from causalrca_codex.core.component import infer_component_type


class FaultIdentificationAgent(BaseAgent):
    """Agent 3: 故障识别Agent - 跨层粗过滤（技术方案 Step 3）。

    职责：
    1. 按组件类型分层：pod/db/redis/service/node/os
    2. 选择主分析层 L*：最大严重度的层
    3. 精炼候选集 C_refined = L* 层的异常组件
    4. 分支决策：
       - |C_refined| = 1 且 score >= tau_single: 直接输出根因，跳过因果推理
       - |C_refined| > 1: 需要因果推理

    层级体系：
    - node/os: 基础设施层
    - pod/docker/container: 容器层
    - service/db/redis/middleware: 服务层
    - app: 应用层
    """

    name = "FaultIdentificationAgent"
    purpose = "跨层粗过滤：按基础设施层分组 + 单组件短路决策"
    preconditions = ["association_layer.candidate_set", "association_layer.anomaly_scores"]
    produces = ["fault_id_layer.refined_candidates", "fault_id_layer.reserve_candidates", "fault_id_layer.needs_causal_inference"]
    tunable_params = {"force_multi_component": False, "restore_reserve": False, "tau_single": 0.8}

    def _execute(self, workspace: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        # ============================================================
        # 步骤目的: 跨层粗过滤（按基础设施层分组 + 主层选择 + 单组件短路决策）
        # 计算方法: ①infer_component_type 推断每组件类型 ②按层聚合 max 分数选主层
        #          ③主层全保留 + 其他层中分数>=50%主层max的组件保留
        # 读取数据: association_layer.candidate_set(异常候选)
        #          association_layer.anomaly_scores(每组件异常分数)
        # ============================================================
        query = workspace["task"]["query"]
        candidates: List[str] = list(workspace["association_layer"].get("candidate_set", []))
        scores: Dict[str, float] = dict(workspace["association_layer"].get("anomaly_scores", {}))
        profiles: Dict[str, Dict[str, Any]] = dict(workspace["association_layer"].get("component_profiles", {}))
        if bool(params.get("restore_reserve")):
            candidates = list(dict.fromkeys(candidates + workspace["fault_id_layer"].get("reserve_candidates", [])))

        evidence_scores: Dict[str, float] = {}
        for component in candidates:
            profile = profiles.get(component, {})
            severity = float(scores.get(component, 0.0))
            local = float(profile.get("local_root_evidence", severity))
            evidence_scores[component] = (
                0.65 * severity
                + 0.35 * local
            )

        # ----- Step 1: 组件分层（按名称推断 type） -----
        print(f"    [FaultIdAgent] 步骤1: 组件分层（读取 association_layer.candidate_set={len(candidates)} 个组件）")
        component_to_layer: Dict[str, str] = {}
        for component in candidates:
            component_to_layer[component] = infer_component_type(component)
        layer_components: Dict[str, List[str]] = defaultdict(list)
        for component, layer in component_to_layer.items():
            layer_components[layer].append(component)
        # 打印分层结果
        for layer, comps in layer_components.items():
            print(f"      层[{layer}]: {len(comps)}个 -> {comps}")

        # ----- Step 2: 每层 max 异常分数聚合 -----
        print(f"    [FaultIdAgent] 步骤2: 各层 max(异常分数) 聚合")
        layer_scores: Dict[str, float] = defaultdict(float)
        for component in candidates:
            layer = component_to_layer[component]
            layer_scores[layer] = max(layer_scores[layer], float(evidence_scores.get(component, scores.get(component, 0.0))))
        for layer, ls in sorted(layer_scores.items(), key=lambda x: -x[1]):
            comp_in_layer = layer_components.get(layer, [])
            comp_with_max = max(comp_in_layer, key=lambda c: evidence_scores.get(c, scores.get(c, 0.0)), default=None)
            print(f"      层[{layer}]: max_score={ls:.4f}  代表组件={comp_with_max}")

        if not candidates:
            refined: List[str] = []
            reserve: List[str] = []
            primary_layer = None
        elif bool(params.get("force_multi_component")):
            refined = candidates
            reserve = []
            primary_layer = "mixed"
        else:
            # ----- Step 3: 主层选择（按 max 分数） -----
            primary_layer = max(layer_scores.items(), key=lambda item: item[1])[0]
            primary_max_score = layer_scores[primary_layer]
            score_threshold = primary_max_score * 0.58
            print(f"    [FaultIdAgent] 步骤3: 主层选择 -> '{primary_layer}' (max_score={primary_max_score:.4f})")
            print(f"    [FaultIdAgent] 步骤3: 其他层保留阈值 = {primary_max_score:.4f} * 0.5 = {score_threshold:.4f}")
            # 主层全保留 + 其他层中分数>=50%主层max的组件
            primary_components = [c for c in candidates if component_to_layer[c] == primary_layer]
            other_components = [c for c in candidates if component_to_layer[c] != primary_layer]
            kept_primary = [
                c for c in primary_components
                if evidence_scores.get(c, 0.0) >= max(score_threshold, primary_max_score * 0.50)
            ]
            if not kept_primary and primary_components:
                kept_primary = [max(primary_components, key=lambda c: evidence_scores.get(c, 0.0))]
            kept_others = [c for c in other_components if evidence_scores.get(c, 0.0) >= score_threshold]
            # Always keep the best from each layer that has meaningful score
            # This prevents losing the true root cause when it's in a different layer
            for layer, layer_score in layer_scores.items():
                if layer == primary_layer or layer_score < primary_max_score * 0.50:
                    continue
                layer_best = max(layer_components.get(layer, []), key=lambda c: evidence_scores.get(c, 0.0), default=None)
                if layer_best and layer_best not in kept_others:
                    kept_others.append(layer_best)
            refined = kept_primary + kept_others
            max_width = max(self.config.top_k * 3, int(getattr(query, "failure_count", 1) or 1) * 4, 8)
            refined = sorted(dict.fromkeys(refined), key=lambda component: evidence_scores.get(component, scores.get(component, 0.0)), reverse=True)[:max_width]
            reserve = [c for c in other_components if c not in kept_others]
            print(f"    [FaultIdAgent] 步骤3: 主层[{primary_layer}]保留={len(primary_components)} 其他层保留={len(kept_others)} 保留池={len(reserve)}")

        refined = sorted(refined, key=lambda component: evidence_scores.get(component, scores.get(component, 0.0)), reverse=True)
        reserve = sorted([c for c in candidates if c not in set(refined)], key=lambda component: evidence_scores.get(component, scores.get(component, 0.0)), reverse=True)

        tau_single = float(params.get("tau_single", self.config.tau_single))
        force_multi = bool(params.get("force_multi_component"))
        # ----- Step 4: 单组件短路决策 -----
        if len(refined) == 1 and not force_multi:
            confidence_single = float(scores.get(refined[0], 0.0))
            needs_causal = confidence_single < tau_single
            tentative = refined[0] if not needs_causal else None
            print(f"    [FaultIdAgent] 步骤4: 单组件短路判断 confidence={confidence_single:.4f} vs tau_single={tau_single:.4f} -> 短路={tentative is not None}")
        else:
            confidence_single = 0.0
            needs_causal = True
            tentative = None
            print(f"    [FaultIdAgent] 步骤4: |refined|={len(refined)} > 1, 强制需要因果推理")

        workspace["fault_id_layer"].update(
            {
                "refined_candidates": refined,
                "reserve_candidates": reserve,
                "primary_layer": primary_layer,
                "needs_causal_inference": needs_causal,
                "tentative_root_cause": tentative,
                "confidence_single": confidence_single,
                "candidate_scores": {component: round(score, 6) for component, score in evidence_scores.items()},
            }
        )

        # 醒目打印故障识别结果
        print(f"    [FaultIdAgent] === 故障识别最终结果 ===")
        print(f"    [FaultIdAgent] 主层={primary_layer} 精炼={len(refined)} 保留={len(reserve)}")
        print(f"    [FaultIdAgent] 需因果推理={needs_causal} 单组件短路={'是' if tentative else '否'}")
        print(f"    [FaultIdAgent] --- 精炼集合 refined({len(refined)}) ---")
        for comp in refined:
            comp_type = component_to_layer.get(comp, infer_component_type(comp))
            print(f"      精炼: {comp} (层={comp_type}, evidence={evidence_scores.get(comp, 0):.4f})")
        if reserve:
            print(f"    [FaultIdAgent] --- 保留池 reserve({len(reserve)}) ---")
            for comp in reserve:
                comp_type = component_to_layer.get(comp, infer_component_type(comp))
                print(f"      保留: {comp} (层={comp_type}, evidence={evidence_scores.get(comp, 0):.4f})")

        return {
            "refined_candidates": refined,
            "reserve_candidates": reserve,
            "primary_layer": primary_layer,
            "needs_causal_inference": needs_causal,
            "tentative_root_cause": tentative,
            "layer_scores": dict(layer_scores),
            "candidate_scores": {component: round(score, 6) for component, score in evidence_scores.items()},
            "confidence_single": confidence_single,
        }

    def _self_evaluate(
        self,
        result: Dict[str, Any],
        workspace: Dict[str, Any],
        params: Dict[str, Any],
    ) -> Tuple[float, List[str]]:
        refined = result["refined_candidates"]
        reserve = result["reserve_candidates"]
        warnings: List[str] = []
        if not refined:
            return 0.10, ["No refined candidate remains; coarse filtering may have removed true root cause."]
        if len(refined) == 1 and result["confidence_single"] >= self.config.tau_single:
            return 0.92, []
        if reserve and len(refined) <= 3:
            warnings.append("Reserve pool is non-empty; keep it available for recovery if downstream evidence is weak.")
        if len(refined) <= 10:
            return 0.85, warnings
        warnings.append("Refined candidate set is still large.")
        return 0.65, warnings
