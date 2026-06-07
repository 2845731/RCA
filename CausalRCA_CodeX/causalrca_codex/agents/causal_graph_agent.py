from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from causalrca_codex.agents.base import BaseAgent
from causalrca_codex.core.component import infer_component_type
from causalrca_codex.core.graph_ops import (
    build_call_counts,
    combined_corr_scores,
    granger_causality_score,
    heuristic_domain_score,
    infer_missing_dependencies,
    lagged_corr_score,
    select_graph_nodes,
    time_precedence_score,
    trace_reliability,
    type_prior_score,
)
from causalrca_codex.llm import LLMClient
from causalrca_codex.prompts import CAUSAL_EDGE_BATCH_PROMPT, CAUSAL_EDGE_PROMPT
from causalrca_codex.schemas import WeightedCausalGraph


class CausalGraphAgent(BaseAgent):
    """Agent 4: 因果图构建Agent（技术方案 Step 4）。

    职责：构建精炼候选组件之间的加权有向因果图 G_causal。
    核心步骤：
    1. 从全天trace中提取调用图 G_call（caller -> callee）
    2. 反转边方向得到因果图骨架（callee -> caller，故障传播方向）
    3. 使用 Scheme-F 混合传播模型计算边权重
    4. 可选图扩展（expand_mode: none/direct/full_path）

    Scheme-F 四因子融合：
    - sA: 时间序证据（原因应先于结果）
    - sB: 滞后Pearson相关性
    - sC: LLM领域知识（可选）
    - sD: 组件类型先验概率
    """

    name = "CausalGraphAgent"
    purpose = "构建加权有向因果图：反转调用图边方向 + Scheme-F混合传播模型计算边权"
    preconditions = ["fault_id_layer.refined_candidates", "association_layer.anomaly_scores"]
    produces = ["causal_graph_layer.weighted_causal_graph", "causal_graph_layer.edge_scores"]
    tunable_params = {"expand_mode": "none"}
    estimated_cost = "medium"

    def _execute(self, workspace: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        # ============================================================
        # 步骤目的: 构建加权有向因果图 G_causal
        # 计算方法: ①从全天 trace 提取调用图 G_call (caller->callee)
        #          ②反转边方向(callee->caller) 表征故障传播方向
        #          ③Scheme-F 四因子计算每条边权重
        #            s_A时间序 + s_B滞后相关 + s_C领域(LLM) + s_D类型先验
        #          ④可选 expand_mode: none/direct/full_path 决定选哪些节点入图
        # 读取数据: data_layer.raw_traces(trace调用关系)
        #          data_layer.raw_metrics(KPI时序)
        #          fault_id_layer.refined_candidates(精炼候选)
        #          association_layer.candidate_set/anomaly_scores(异常候选+分数)
        # ============================================================
        expand_mode = str(params.get("expand_mode", self.config.expand_mode))
        refined = list(workspace["fault_id_layer"].get("refined_candidates", []))
        anomalous = list(workspace["association_layer"].get("candidate_set", []))
        trace_frames = workspace["data_layer"].get("raw_traces", [])
        query = workspace["task"]["query"]
        print(f"    [CausalGraphAgent] 读取 data_layer.raw_traces={len(trace_frames)} trace文件")
        print(f"    [CausalGraphAgent] 读取 fault_id_layer.refined_candidates={len(refined)} 个精炼组件")
        print(f"    [CausalGraphAgent] expand_mode={expand_mode} | 边权重公式: w=r_trace*s_F + (1-r_trace)*s_prior")
        print(f"    [CausalGraphAgent] Scheme-F五因子: sA(时间序) + sB(相关性) + sE(Granger因果) + sD(类型先验)")

        # Step 4.1: 从全天trace提取调用关系
        call_counts = build_call_counts(trace_frames, query)
        print(f"    [CausalGraphAgent] 步骤1: trace提取调用对数={len(call_counts)}")

        # Infer missing dependencies for db/redis/node components
        # (trace data often doesn't include database spans)
        call_counts = infer_missing_dependencies(call_counts, refined, query)
        print(f"    [CausalGraphAgent] 步骤1补: 缺失依赖推断后调用对数={len(call_counts)}")

        # Step 4.3: 根据expand_mode选择图节点
        selected_nodes = select_graph_nodes(call_counts, refined, anomalous, expand_mode)
        if not selected_nodes:
            selected_nodes = set(refined or anomalous)
        print(f"    [CausalGraphAgent] 步骤2: 选中节点数={len(selected_nodes)}")

        graph = WeightedCausalGraph()
        anomaly_scores = workspace["association_layer"].get("anomaly_scores", {})
        intensity = workspace["association_layer"].get("component_intensity_series", {})
        first_ts = workspace["association_layer"].get("first_anomaly_ts", {})
        anomaly_details = workspace["association_layer"].get("anomaly_details", {})

        # Innovation: Multi-resolution onset time for temporal discrimination.
        # Use the finest-resolution onset time (1min scale) for each component.
        # This gives much better temporal ordering than the 30-minute window.
        multi_res_onset = workspace["association_layer"].get("multi_resolution_onset", {})
        per_kpi_onset = workspace["association_layer"].get("per_kpi_onset", {})
        onset_ts = {}
        for comp in selected_nodes:
            # Priority 1: Multi-resolution 1-minute onset (finest scale)
            comp_res = multi_res_onset.get(comp, {})
            if "60s" in comp_res:
                onset_ts[comp] = comp_res["60s"]
            elif "300s" in comp_res:
                onset_ts[comp] = comp_res["300s"]
            elif "900s" in comp_res:
                onset_ts[comp] = comp_res["900s"]
            else:
                # Priority 2: Per-KPI onset
                kpi_times = per_kpi_onset.get(comp, {})
                if kpi_times:
                    onset_ts[comp] = min(kpi_times.values())
                else:
                    # Priority 3: Segment-based onset
                    details = anomaly_details.get(comp, [])
                    earliest = None
                    for seg in details:
                        points = seg.get("points", [])
                        if points:
                            for pt in points:
                                if float(pt.get("deviation", 0.0)) >= 0.3:
                                    ts = int(pt.get("timestamp", 0))
                                    if earliest is None or ts < earliest:
                                        earliest = ts
                                    break
                        elif earliest is None:
                            earliest = int(seg.get("start_ts", 0))
                    onset_ts[comp] = earliest if earliest is not None else first_ts.get(comp)

        # 添加节点（附带severity、时间、KPI、类型属性）
        for node in selected_nodes:
            graph.add_node(
                node,
                component_type=infer_component_type(node),
                anomalous=node in anomalous,
                severity=float(anomaly_scores.get(node, 0.0)),
            )

        # Step 4.2 + 4.4: 反转边方向 + Scheme-F计算边权
        # 反转：caller→callee 变为 callee→caller（故障传播方向）
        # 当callee故障时，caller的请求会失败，故障从callee传播到caller。

        # Collect all valid edges for batch LLM scoring
        valid_edges = []
        for (caller, callee), count in sorted(call_counts.items(), key=lambda x: (x[0][0], x[0][1])):
            source = callee
            target = caller
            if source not in selected_nodes or target not in selected_nodes or source == target:
                continue
            valid_edges.append((source, target, count))

        # Batch LLM scoring (one API call for all edges)
        llm_scores = self._batch_domain_scores(valid_edges, first_ts)

        edge_scores = {}
        for source, target, count in valid_edges:
            # Scheme-F 五因子（创新：加入Granger因果性 sE）
            s_time = time_precedence_score(onset_ts.get(source), onset_ts.get(target), self.config)
            s_corr, s_granger = combined_corr_scores(intensity.get(source), intensity.get(target))
            s_llm = llm_scores.get(f"{source}->{target}")
            s_prior = type_prior_score(source, target, self.config)

            # 融合权重（创新：五因子融合公式）
            if self.config.use_llm_edge_scoring and s_llm is not None:
                s_f = 0.20 * s_time + 0.20 * s_corr + 0.20 * s_granger + 0.20 * s_llm + 0.20 * s_prior
            else:
                s_f = (
                    self.config.alpha_time * s_time
                    + self.config.alpha_corr * s_corr
                    + self.config.alpha_granger * s_granger
                    + self.config.alpha_type_prior * s_prior
                )

            # 冲突调整（技术方案一致性检查）
            if self.config.use_llm_edge_scoring and s_llm is not None:
                if s_llm > 0.7 and s_time < 0.2:
                    s_f *= 0.6
                elif s_llm < 0.3 and s_corr > 0.8:
                    s_f = min(1.0, s_f * 1.3)

            r_trace = trace_reliability(count, self.config)
            weight = max(0.0, min(1.0, r_trace * s_f + (1.0 - r_trace) * s_prior))

            scores = {
                "s_A_time": round(s_time, 6),
                "s_B_corr": round(s_corr, 6),
                "s_E_granger": round(s_granger, 6),
                "s_C_domain": round(s_llm, 6) if s_llm is not None else None,
                "s_D_type_prior": round(s_prior, 6),
                "r_trace": round(r_trace, 6),
                "s_F": round(s_f, 6),
            }
            graph.add_edge(source, target, weight=round(weight, 6), call_count=count, scores=scores)
            edge_detail = dict(scores)
            edge_detail.update({"weight": round(weight, 6), "call_count": count})
            edge_scores[f"{source}->{target}"] = edge_detail

        # 无边时添加孤立节点
        if not graph.edges and selected_nodes:
            for node in selected_nodes:
                graph.add_node(node, component_type=infer_component_type(node), anomalous=node in anomalous)

        # Step 4.5: 写入causal_graph_layer
        workspace["causal_graph_layer"].update(
            {
                "local_call_graph": {f"{caller}->{callee}": count for (caller, callee), count in call_counts.items()},
                "weighted_causal_graph": graph,
                "edge_scores": edge_scores,
                "graph_config": {"expand_mode": expand_mode},
            }
        )

        # ============================================================
        # 保存因果图到独立文件夹（多个图）
        # ============================================================
        output_root = Path(self.config.resolved_output_root()) if hasattr(self.config, "resolved_output_root") else None
        graph_save_dir: Path = None  # type: ignore
        if output_root is not None:
            case_dir = output_root / query.dataset / f"row_{query.row_id}_causal_graphs"
            case_dir.mkdir(parents=True, exist_ok=True)
            graph_save_dir = case_dir
            # 图1：调用图骨架（原始 trace，caller->callee，权重=call_count）
            call_graph = {f"{c}->{ca}": cnt for (c, ca), cnt in call_counts.items()}
            # 图2：因果图（反转方向，权重=Scheme-F 融合权重）
            causal_edges = [
                {
                    "source": e.source,
                    "target": e.target,
                    "weight": e.weight,
                    "call_count": e.call_count,
                    "scores": e.scores,
                }
                for e in graph.edges
            ]
            causal_graph_data = {
                "figure_name": "图2-因果图(反转+Scheme-F)",
                "direction_meaning": "source->target 表示故障从source传播到target",
                "nodes": [
                    {
                        "component": n,
                        "component_type": graph.nodes[n].get("component_type"),
                        "anomalous": graph.nodes[n].get("anomalous"),
                        "severity": graph.nodes[n].get("severity"),
                    }
                    for n in graph.nodes
                ],
                "edges": causal_edges,
            }
            call_graph_data = {
                "figure_name": "图1-调用图骨架(原始trace, caller->callee)",
                "edges": [{"caller": k.split("->")[0], "callee": k.split("->")[1], "call_count": v} for k, v in call_graph.items()],
            }
            try:
                with (case_dir / "graph1_call_skeleton.json").open("w", encoding="utf-8") as f:
                    json.dump(call_graph_data, f, ensure_ascii=False, indent=2)
                with (case_dir / "graph2_causal.json").open("w", encoding="utf-8") as f:
                    json.dump(causal_graph_data, f, ensure_ascii=False, indent=2)
                # 同时输出可读的 .txt
                with (case_dir / "graph2_causal_readable.txt").open("w", encoding="utf-8") as f:
                    f.write(f"=== 图2: 因果图 (callee->caller, 故障传播方向) ===\n")
                    f.write(f"节点数: {len(graph.nodes)} 边数: {len(graph.edges)}\n")
                    f.write(f"expand_mode: {expand_mode}\n\n")
                    f.write("--- 节点 ---\n")
                    for n in graph.nodes:
                        nd = graph.nodes[n]
                        f.write(f"  {n} (type={nd.get('component_type')}, anomalous={nd.get('anomalous')}, severity={nd.get('severity'):.4f})\n")
                    f.write("\n--- 边 (源=被调->目标=主调, 故障从源传到目标) ---\n")
                    for e in sorted(graph.edges, key=lambda x: -x.weight):
                        f.write(f"  {e.source}({infer_component_type(e.source)}) --(w={e.weight:.4f}, calls={e.call_count})--> {e.target}({infer_component_type(e.target)})\n")
            except Exception as exc:
                print(f"    [CausalGraphAgent] ⚠️ 保存因果图失败: {exc}")

        # ============================================================
        # 醒目打印图构建结果
        # ============================================================
        print(f"    [CausalGraphAgent] === 因果图构建结果 ===")
        print(f"    [CausalGraphAgent] 节点={len(graph.nodes)} 边={len(graph.edges)} expand_mode={expand_mode}")
        if graph_save_dir is not None:
            print(f"    [CausalGraphAgent] 因果图已保存到文件夹: {graph_save_dir}")
        # 图1: 调用图骨架
        print(f"    [CausalGraphAgent] --- 图1: 调用图骨架 (caller->callee, 权重=call_count) ---")
        sorted_call = sorted(call_counts.items(), key=lambda x: -x[1])
        for i, ((caller, callee), cnt) in enumerate(sorted_call, 1):
            print(f"      #{i:>2} {caller}({infer_component_type(caller)}) --(calls={cnt})--> {callee}({infer_component_type(callee)})")
        # 图2: 因果图
        print(f"    [CausalGraphAgent] --- 图2: 因果图 (callee->caller, 故障传播方向, 权重=Scheme-F) ---")
        if graph.edges:
            for i, edge in enumerate(sorted(graph.edges, key=lambda x: -x.weight), 1):
                src_type = infer_component_type(edge.source)
                tgt_type = infer_component_type(edge.target)
                s = edge.scores or {}
                print(f"      #{i:>2} {edge.source}({src_type}) --(w={edge.weight:.4f}, calls={edge.call_count}, sA={s.get('s_A_time',0):.3f}, sB={s.get('s_B_corr',0):.3f}, sE={s.get('s_E_granger',0):.3f}, sD={s.get('s_D_type_prior',0):.3f})--> {edge.target}({tgt_type})")
        else:
            print(f"      (无因果边)")

        return {
            "nodes": list(graph.nodes.keys()),
            "edges": [edge.to_dict() for edge in graph.edges],
            "edge_scores": edge_scores,
            "expand_mode": expand_mode,
        }

    def _self_evaluate(
        self,
        result: Dict[str, Any],
        workspace: Dict[str, Any],
        params: Dict[str, Any],
    ) -> Tuple[float, List[str]]:
        nodes = result["nodes"]
        edges = result["edges"]
        refined = workspace["fault_id_layer"].get("refined_candidates", [])
        warnings: List[str] = []
        if not nodes:
            return 0.10, ["No graph nodes could be constructed."]
        coverage = sum(1 for component in refined if component in nodes) / max(1, len(refined))
        score = 0.35 + 0.35 * coverage
        if edges:
            avg_weight = sum(edge["weight"] for edge in edges) / len(edges)
            # Scale edge quality bonus by edge count (more edges = better connectivity)
            edge_bonus = min(0.30, 0.10 + 0.05 * len(edges))
            score += edge_bonus * min(1.0, avg_weight)
            # Check for causal source (node with in-degree 0)
            has_source = any(
                not any(e["target"] == n for e in edges)
                for n in nodes
            )
            if not has_source:
                warnings.append("No causal source found; graph may have cycles.")
                score *= 0.9
        else:
            warnings.append("Weighted causal graph has no edges; expand_mode should be increased if intervention confidence is low.")
            score *= 0.5
        return min(1.0, score), warnings

    def _domain_score(self, source: str, target: str, call_count: int, first_ts: Dict[str, Any]) -> float:
        fallback = heuristic_domain_score(source, target, edge_is_trace_backed=call_count > 0)
        if not self.config.use_llm_edge_scoring:
            return fallback
        client = LLMClient(self.config)
        payload = client.complete_json(
            CAUSAL_EDGE_PROMPT,
            "\n".join(
                [
                    f"Component X: {source}",
                    f"Component Y: {target}",
                    f"Trace relation: {target} calls {source}; propagation direction is {source} -> {target}.",
                    f"Call count: {call_count}",
                    f"X first anomaly timestamp: {first_ts.get(source)}",
                    f"Y first anomaly timestamp: {first_ts.get(target)}",
                ]
            ),
        )
        if not payload:
            return fallback
        try:
            return max(0.0, min(1.0, float(payload.get("propagation_probability", fallback))))
        except Exception:
            return fallback

    def _batch_domain_scores(
        self, edges: list, first_ts: Dict[str, Any]
    ) -> Dict[str, float]:
        """Batch score all edges with a single LLM call."""
        fallbacks = {
            f"{s}->{t}": heuristic_domain_score(s, t, edge_is_trace_backed=c > 0)
            for s, t, c in edges
        }
        if not self.config.use_llm_edge_scoring or not edges:
            return fallbacks

        from causalrca_codex.prompts import CAUSAL_EDGE_BATCH_PROMPT

        edge_descriptions = []
        for i, (source, target, call_count) in enumerate(edges):
            edge_descriptions.append(
                f"[{i}] {source} -> {target} (calls={call_count}, "
                f"X_ts={first_ts.get(source)}, Y_ts={first_ts.get(target)})"
            )

        client = LLMClient(self.config)
        payload = client.complete_json(
            CAUSAL_EDGE_BATCH_PROMPT,
            "Edges to score:\n" + "\n".join(edge_descriptions),
        )

        if not payload or not isinstance(payload, list):
            return fallbacks

        results = dict(fallbacks)
        for item in payload:
            try:
                idx = int(item.get("edge_id", -1))
                if 0 <= idx < len(edges):
                    key = f"{edges[idx][0]}->{edges[idx][1]}"
                    val = float(item.get("propagation_probability", fallbacks[key]))
                    results[key] = max(0.0, min(1.0, val))
            except (ValueError, TypeError, KeyError):
                continue
        return results
