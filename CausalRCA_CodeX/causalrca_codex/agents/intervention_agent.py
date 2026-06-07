from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from causalrca_codex.agents.base import BaseAgent
from causalrca_codex.core.graph_ops import max_path_strengths
from causalrca_codex.schemas import WeightedCausalGraph


class InterventionAgent(BaseAgent):
    """Agent 5: 干预Agent - ExplainScore排名（技术方案 Step 5, Pearl Level 2）。

    职责：计算每个候选组件的ExplainScore(ES)，衡量其因果影响力覆盖度。

    核心概念 - InfluenceScore（瓶颈模型）：
    替代朴素乘积模型。乘积模型随路径长度单调衰减，隐含"故障传播减弱"假设，
    但现实中存在放大效应（重试风暴、级联故障）。瓶颈模型取所有路径中
    最小边权的最大值：
        InfScore(X->Y) = max_p [ min_e∈p w(e) ]

    ES公式：
        ES(X) = Σ sev(Y) * InfScore(X→Y) / Σ sev(Y)
    其中 D_X 是X的异常下游组件集合。

    排名公式：多因子加权
        RCS = 0.25*severity + 0.10*early + 0.20*type_prior + 0.15*ES
            + 0.10*diversity + 0.10*latency + 0.10*source_bonus

    type_prior编码SRE领域知识：service > pod > node > middleware > redis > db
    """

    name = "InterventionAgent"
    purpose = "干预评分：瓶颈InfluenceScore + ExplainScore排名"
    preconditions = ["causal_graph_layer.weighted_causal_graph", "association_layer.anomaly_scores"]
    produces = ["intervention_layer.explain_scores", "intervention_layer.root_cause_scores", "intervention_layer.ranking"]
    estimated_cost = "medium"

    def _execute(self, workspace: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        # ============================================================
        # 步骤目的: 干预评分(Pearl Level 2 do-calculus) - 评估"反事实"移除节点效果
        # 计算方法: ①ES(ExplainScore) = max-min瓶颈路径算法
        #            衡量移除某节点后对其他节点的最大影响衰减
        #          ②RCS(RootCauseScore) = eta_self*severity + eta_early*time + eta_source*in_degree_zero
        #            自我严重度 + 时间越早 + 因果源(in_degree=0) 越像根因
        # 读取数据: causal_graph_layer.weighted_causal_graph(Scheme-F因果图)
        #          association_layer.anomaly_scores(异常分数)
        #          association_layer.first_anomaly_ts(各组件首次异常时间)
        #          fault_id_layer.refined_candidates(精炼候选集)
        # ============================================================
        graph = workspace["causal_graph_layer"].get("weighted_causal_graph")
        anomaly_scores: Dict[str, float] = workspace["association_layer"].get("anomaly_scores", {})
        anomaly_details: Dict[str, List[Dict[str, Any]]] = workspace["association_layer"].get("anomaly_details", {})
        refined = list(workspace["fault_id_layer"].get("refined_candidates", []))
        tentative = workspace["fault_id_layer"].get("tentative_root_cause")
        first_ts = workspace["association_layer"].get("first_anomaly_ts", {})
        print(f"    [InterventionAgent] 读取 causal_graph_layer 节点={len(graph.nodes) if graph else 0} 边={len(graph.edges) if graph else 0}")
        print(f"    [InterventionAgent] 读取 fault_id_layer.refined_candidates={len(refined)} 个精炼组件")
        print(f"    [InterventionAgent] ES算法: max-min瓶颈路径, RCS公式: 0.40*ES + 0.20*type + 0.12*early + 0.08*sev + 0.08*div + 0.07*src + 0.05*prop")

        candidates = refined or list(anomaly_scores.keys())
        if tentative and tentative not in candidates:
            candidates.insert(0, tentative)

        explain_scores: Dict[str, float] = {}
        root_scores: Dict[str, float] = {}
        path_strengths_by_candidate: Dict[str, Dict[str, float]] = {}

        if graph is None:
            graph = WeightedCausalGraph()
            for component in candidates:
                graph.add_node(component, severity=float(anomaly_scores.get(component, 0.0)))

        t_min = min([ts for ts in first_ts.values() if ts is not None], default=None)

        # Compute per-component first anomaly segment start time.
        # This gives a more granular temporal signal than the shared window-based first_ts,
        # because different components' earliest anomaly segments start at different times.
        segment_first_ts: Dict[str, Optional[float]] = {}
        for comp in candidates:
            details = anomaly_details.get(comp, [])
            earliest = None
            for seg in details:
                start = seg.get("start_ts")
                if start is not None and (earliest is None or start < earliest):
                    earliest = start
            segment_first_ts[comp] = earliest

        # Compute per-component latency score from trace span durations
        # True root causes typically have higher span durations
        latency_stats = workspace.get("data_layer", {}).get("component_latency_stats", {})
        max_avg_latency = max(
            (s["avg_duration"] for s in latency_stats.values()), default=1.0
        )
        latency_scores: Dict[str, float] = {}
        for comp in candidates:
            stats = latency_stats.get(comp)
            if stats and max_avg_latency > 0:
                latency_scores[comp] = min(1.0, stats["avg_duration"] / max_avg_latency)
            else:
                latency_scores[comp] = 0.5  # Unknown → neutral

        # Compute anomaly diversity score: true root causes typically show
        # multiple distinct KPI types (e.g., CPU + memory + network),
        # while downstream effects often show fewer types.
        all_hints = set()
        hints_by_component: Dict[str, set] = {}
        for comp, details in anomaly_details.items():
            hints = set()
            for seg in details:
                hint = seg.get("reason_hint")
                if hint:
                    hints.add(hint)
                    all_hints.add(hint)
            hints_by_component[comp] = hints
        max_hints = max(len(all_hints), 1)
        diversity_scores: Dict[str, float] = {}
        for comp in candidates:
            n_hints = len(hints_by_component.get(comp, set()))
            diversity_scores[comp] = math.log2(1 + n_hints) / math.log2(1 + max_hints) if max_hints > 0 else 0.0

        from causalrca_codex.core.component import infer_component_type

        for candidate in candidates:
            if not graph.has_node(candidate):
                graph.add_node(candidate, anomalous=candidate in anomaly_scores, severity=float(anomaly_scores.get(candidate, 0.0)))
            strengths = max_path_strengths(graph, candidate, max_depth=self.config.max_path_depth)
            path_strengths_by_candidate[candidate] = strengths
            # ES(X) = Σ_{Y in D_X} sev(Y) * InfScore(X→Y) / Σ sev(Y)
            # D_X = downstream anomalous nodes (not including X itself)
            # Innovation: Enhanced competing-cause correction — reduce ES when
            # multiple candidates explain the same downstream anomaly.
            # This prevents hub components from getting inflated ES.
            explain = 0.0
            downstream_sev = 0.0
            for target, severity in anomaly_scores.items():
                if target == candidate:
                    continue  # Exclude self from ES calculation
                strength = strengths.get(target, 0.0)
                if strength > 0:
                    # Count how many other candidates also explain this target
                    other_explainers = sum(
                        1 for other in candidates
                        if other != candidate
                        and path_strengths_by_candidate.get(other, {}).get(target, 0.0) > 0.1
                    )
                    # Soft correction: reduce by up to 30% based on competition
                    competition_factor = max(0.7, 1.0 - 0.075 * other_explainers)
                    explain += float(severity) * strength * competition_factor
                    downstream_sev += float(severity)
            if downstream_sev > 0:
                explain = explain / downstream_sev
            else:
                # No downstream anomalies reachable. Check if node has any outgoing edges
                # in the graph. If it's truly isolated (no edges at all), it's likely a
                # data gap rather than a true leaf, so use moderate score.
                # If it has outgoing edges but none reach anomalous nodes, it's a true
                # non-explainer, so penalize more.
                has_outgoing = len(list(graph.outgoing(candidate))) > 0
                max_sev = max(anomaly_scores.values()) if anomaly_scores else 1.0
                raw = float(anomaly_scores.get(candidate, 0.0)) / max_sev if max_sev > 0 else 0.0
                if has_outgoing:
                    explain = raw * 0.3  # Has edges but doesn't explain anomalies
                else:
                    explain = raw * 0.6  # Truly isolated - might be data gap
            explain_scores[candidate] = round(max(0.0, min(1.0, explain)), 6)

            # Propagation score: boost components with strong outgoing edges to
            # anomalous downstream nodes. Root causes propagate faults outward.
            outgoing_edges = graph.outgoing(candidate)
            weighted_outgoing = sum(
                float(getattr(e, 'weight', 0.5))
                for e in outgoing_edges
                if e.target in anomaly_scores
            )
            total_outgoing_weight = sum(float(getattr(e, 'weight', 0.5)) for e in outgoing_edges) or 1.0
            propagation_score = min(1.0, weighted_outgoing / total_outgoing_weight)

            # Incoming edge penalty: compare outgoing-to-anomalous vs incoming-from-anomalous.
            # True root causes: high outgoing-to-anomalous, low incoming-from-anomalous.
            # Hub/downstream: high both. Penalize when incoming-from-anomalous is high.
            incoming_edges = graph.incoming(candidate)
            weighted_incoming = sum(
                float(getattr(e, 'weight', 0.5))
                for e in incoming_edges
                if e.source in anomaly_scores
            )
            total_incoming_weight = sum(float(getattr(e, 'weight', 0.5)) for e in incoming_edges) or 1.0
            incoming_ratio = weighted_incoming / total_incoming_weight
            # Root cause signal: outgoing-to-anomalous is high, incoming-from-anomalous is low
            root_cause_signal = propagation_score * (1.0 - 0.5 * incoming_ratio)
            propagation_score = min(1.0, root_cause_signal)

            # Source node bonus: nodes with no incoming edges from anomalous parents
            # are more likely to be root causes (they are the source of fault propagation).
            anomalous_in_degree = sum(1 for e in incoming_edges if e.source in anomaly_scores)
            source_bonus = 1.0 if anomalous_in_degree == 0 else max(0.3, 1.0 - 0.2 * anomalous_in_degree)
            # Use per-component segment start time for early_score (falls back to window-based first_ts)
            comp_first = segment_first_ts.get(candidate) or first_ts.get(candidate)
            t_ref = min([v for v in segment_first_ts.values() if v is not None], default=t_min)
            if t_ref is None or comp_first is None:
                early_score = 0.5
            else:
                early_score = math.exp(-(float(comp_first) - float(t_ref)) / max(self.config.lambda_early, 1e-6))
            self_severity = float(anomaly_scores.get(candidate, 0.0))
            diversity = diversity_scores.get(candidate, 0.0)
            latency = latency_scores.get(candidate, 0.5)

            # Component type prior: moderate differentiation
            # Service components slightly favored, but not overwhelmingly
            comp_type = infer_component_type(candidate)
            type_prior = {
                "service": 0.85,
                "pod": 0.80,
                "node": 0.75,
                "middleware": 0.75,
                "redis": 0.70,
                "db": 0.65,
            }.get(comp_type, 0.70)

            # Innovation: Ensemble ranking with multiple formula variants.
            # Includes type-prior to encode SRE domain knowledge about which
            # component types are more likely to be root causes.
            # Service > pod > middleware > node > redis > db
            #
            # Variant 1: ES-dominant (causal influence is primary)
            rcs_v1 = (
                0.40 * explain_scores[candidate]
                + 0.30 * type_prior
                + 0.15 * early_score
                + 0.15 * self_severity
            )
            # Variant 2: Severity-dominant (direct evidence is primary)
            rcs_v2 = (
                0.10 * explain_scores[candidate]
                + 0.20 * type_prior
                + 0.20 * early_score
                + 0.50 * self_severity
            )
            # Variant 3: Type-prior-dominant (domain knowledge is primary)
            rcs_v3 = (
                0.15 * explain_scores[candidate]
                + 0.50 * type_prior
                + 0.15 * early_score
                + 0.20 * self_severity
            )
            # Blend: average of all three variants
            root_score = (rcs_v1 + rcs_v2 + rcs_v3) / 3.0
            root_scores[candidate] = round(max(0.0, min(1.0, root_score)), 6)

        ranking = [
            {
                "component": component,
                "ExplainScore": explain_scores.get(component, 0.0),
                "RootCauseScore": score,
                "P_approx": 0.0,
                "path_strengths": path_strengths_by_candidate.get(component, {}),
            }
            for component, score in sorted(root_scores.items(), key=lambda item: item[1], reverse=True)
        ]

        probs = self._softmax([row["RootCauseScore"] for row in ranking])
        for row, prob in zip(ranking, probs):
            row["P_approx"] = round(prob, 6)

        top1 = ranking[0]["RootCauseScore"] if ranking else 0.0
        top2 = ranking[1]["RootCauseScore"] if len(ranking) > 1 else 0.0
        graph_quality = float(workspace["causal_graph_layer"].get("quality", 0.0))
        confidence = (
            self.config.delta_score * top1
            + self.config.delta_margin * max(0.0, top1 - top2)
            + self.config.delta_graph_quality * graph_quality
        )
        topk = ranking[: self.config.top_k]

        # 醒目打印排名结果
        print(f"    [InterventionAgent] 候选数={len(ranking)} top1_confidence={confidence:.4f}")
        for i, row in enumerate(ranking[:5]):
            comp = row["component"]
            es = row["ExplainScore"]
            rcs = row["RootCauseScore"]
            sev = float(anomaly_scores.get(comp, 0.0))
            comp_tp = infer_component_type(comp)
            tp = {"service": 0.85, "pod": 0.80, "node": 0.75, "middleware": 0.75, "redis": 0.70, "db": 0.65}.get(comp_tp, 0.70)
            print(f"      #{i+1} {comp}({comp_tp}): ES={es:.4f} RCS={rcs:.4f} sev={sev:.4f} type_prior={tp:.2f}")

        workspace["intervention_layer"].update(
            {
                "explain_scores": explain_scores,
                "root_cause_scores": root_scores,
                "ranking": ranking,
                "top1_confidence": round(confidence, 6),
                "topk_candidates": topk,
            }
        )
        return {
            "explain_scores": explain_scores,
            "root_cause_scores": root_scores,
            "ranking": ranking,
            "top1_confidence": round(confidence, 6),
            "topk_candidates": topk,
        }

    def _softmax(self, values: List[float], kappa: float = 5.0) -> List[float]:
        if not values:
            return []
        shifted = [kappa * (value - max(values)) for value in values]
        exp_values = [math.exp(value) for value in shifted]
        denom = sum(exp_values) or 1.0
        return [value / denom for value in exp_values]

    def _self_evaluate(
        self,
        result: Dict[str, Any],
        workspace: Dict[str, Any],
        params: Dict[str, Any],
    ) -> Tuple[float, List[str]]:
        confidence = float(result["top1_confidence"])
        warnings: List[str] = []
        if not result["ranking"]:
            return 0.10, ["Intervention ranking is empty."]
        if confidence < 0.40:
            warnings.append("Top-1 intervention confidence is below 0.40; graph expansion or edge reweighting is recommended.")
        if len(result["ranking"]) > 1:
            margin = result["ranking"][0]["RootCauseScore"] - result["ranking"][1]["RootCauseScore"]
            if margin < 0.05:
                warnings.append("Top candidates have very similar RootCauseScore values.")
        return max(0.10, min(1.0, confidence)), warnings
