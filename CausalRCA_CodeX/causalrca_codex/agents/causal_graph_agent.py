from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from causalrca_codex.agents.base import BaseAgent
from causalrca_codex.agents.log_topology_agent import LogTopologyAgent
from causalrca_codex.core.component import infer_component_type
from causalrca_codex.core.graph_ops import (
    build_call_counts,
    heuristic_domain_score,
    infer_missing_dependencies,
    lagged_corr_score,
    select_graph_nodes,
    time_precedence_score,
    trace_reliability,
    type_prior_score,
)
from causalrca_codex.llm import LLMClient
from causalrca_codex.prompts import CAUSAL_EDGE_PROMPT
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

        # ====================================================================
        # 数据源分配（与 DataAgent 数据范围规则严格一致）
        # - 图1 调用图骨架：使用全天 trace（raw_traces_full_day）
        #   原因：窗口内 trace 可能不完整（尤其故障刚开始的几分钟内），只用
        #   窗口 trace 会漏掉大量背景调用关系，导致调用图骨架失真。
        # - 图2 因果图边权重/相关性：使用故障窗口内 trace（raw_traces）
        #   原因：仅在故障窗口内的事件才有诊断意义，相关性计算需要贴合
        #   故障时间窗。
        # ====================================================================
        trace_frames_full_day = workspace["data_layer"].get("raw_traces_full_day", [])
        trace_frames_window = workspace["data_layer"].get("raw_traces", [])
        query = workspace["task"]["query"]
        tw = workspace["data_layer"].get("trace_time_windows", {})

        # 输入数据规模统计
        def _frame_rows(df_or_frame):
            """兼容 TelemetryFrame / DataFrame 的行数读取."""
            if hasattr(df_or_frame, "rows"):  # TelemetryFrame
                try:
                    return int(df_or_frame.rows)
                except Exception:
                    return 0
            try:
                return int(len(df_or_frame))
            except Exception:
                return 0

        def _trace_stats(frames, label):
            n_files = len(frames)
            n_rows = int(sum(_frame_rows(df) for df in frames)) if frames else 0
            print(f"    [CausalGraphAgent][输入] {label}: 文件数={n_files}  span行数={n_rows}")
            return n_files, n_rows

        print(f"    [CausalGraphAgent] ★ 数据范围说明：")
        print(f"      - 图1 调用图骨架 → 使用【全天 trace】 (DataAgent.raw_traces_full_day)")
        print(f"      - 图2 因果图     → 使用【故障窗口内 trace】 (DataAgent.raw_traces)")
        print(f"      - 故障时间窗   : [{tw.get('effective_start_ts', '?')}, {tw.get('end_ts', '?')}]  (epoch秒)")
        if query.start_time and query.end_time:
            print(f"      - 人类可读窗口 : {query.start_time}  ~  {query.end_time}")
        print(f"    [CausalGraphAgent] 读取 fault_id_layer.refined_candidates={len(refined)} 个精炼组件")
        print(f"    [CausalGraphAgent] expand_mode={expand_mode} | EdgeWeight = fused topology + telemetry support")

        n_files_day, n_rows_day = _trace_stats(trace_frames_full_day, "[图1] 全天 trace")
        n_files_win, n_rows_win = _trace_stats(trace_frames_window, "[图2] 窗口 trace")

        # --------------------------------------------------------------------
        # 步骤1（图1）: 从全天 trace 提取调用关系，构建调用图骨架
        # 输入: trace_frames_full_day  (List[DataFrame], 每行=一个 span)
        # 列匹配: (trace_id, span_id, parent_id, cmdb_id) 或 (traceId,id,pid,cmdb_id)
        #        或 (cmdb_id, dsName) / (cmdb_id, serviceName)
        # 输出: {(caller, callee): call_count}
        # 目的: 还原完整的组件间调用拓扑，作为因果图（图2）的骨架
        # --------------------------------------------------------------------
        try:
            trace_call_counts = build_call_counts(trace_frames_full_day, query)
        except Exception as exc:
            print(f"    [CausalGraphAgent][错误] build_call_counts 失败: {type(exc).__name__}: {exc}")
            trace_call_counts = {}
        call_counts = dict(trace_call_counts)
        print(f"    [CausalGraphAgent] 步骤1: 全天 trace→ 调用对数={len(call_counts)}, "
              f"输入 span行数={n_rows_day} (来自 {n_files_day} 个文件)")
        top5 = sorted(call_counts.items(), key=lambda kv: -kv[1])[:5]
        if top5:
            print(f"    [CausalGraphAgent] 步骤1 样例: top-5 高频调用对 (caller->callee: count):")
            for (c, ca), cnt in top5:
                print(f"      - {c} -> {ca} : {cnt}")

        # ====================================================================
        # 步骤1补：缺失依赖推断 —— 已禁用
        # 原因：原 infer_missing_dependencies() 按组件类型 (db/redis/node) 推断
        #       "service→db" 全连接边，会引入大量噪声（n×m 条臆测边），
        #       使根因排序时所有 db 看起来都"差不多像根因"，反而稀释了
        #       真实异常路径的区分度。
        # 替代方案：让 expand_mode="full_path" 在 BFS 时基于真实 trace 边
        #           自然补全节点；缺失的边让它"诚实地缺失"。
        # 如需重新启用，请先修复以下两个问题：
        #   1) 推断方向与 docstring 注释不一致（代码写 service→db，注释写 db→service）
        #   2) 完全不知道某个 db 实际被哪些 service 调用，一律假定全部
        # ====================================================================
        # call_counts = infer_missing_dependencies(call_counts, refined, query)
        print(f"    [CausalGraphAgent] 步骤1补: 缺失依赖推断已禁用 (使用真实 trace 边: {len(call_counts)} 条)")

        # ====================================================================
        # 步骤1.5: LogTopologyAgent (日志挖掘的拓扑补充)
        # 从故障窗口 log 中挖掘 service->db/redis 边, 与主图合并。
        # 详细过程由 LogTopologyAgent 自己打印。
        # ====================================================================
        try:
            log_agent = LogTopologyAgent(self.config)
            # 注意: 此时 causal_graph_layer 还没建立, LogTopologyAgent 会去
            # data_layer/local_call_graph / 都没找到时, 认为主图边数 = 0
            # 我们手动注入一个临时字段, 让 LogTopologyAgent 能读到主图
            workspace.setdefault("data_layer", {})["local_call_graph"] = {
                f"{k[0]}->{k[1]}": v for k, v in call_counts.items()
            }
            log_agent._execute(workspace, params)
        except Exception as exc:
            print(f"    [CausalGraphAgent][错误] LogTopologyAgent 失败: {type(exc).__name__}: {exc}")
            workspace.setdefault("log_topology_layer", {})
            workspace["log_topology_layer"]["call_counts"] = {}
            workspace["log_topology_layer"]["status"] = "FAILED"
        # 取出挖掘结果
        log_layer = workspace.get("log_topology_layer", {})
        log_call_counts = log_layer.get("call_counts", {})
        log_status = log_layer.get("status", "EMPTY")
        log_max_r = float(log_layer.get("max_r_trace", 0.55))

        # 合并策略: max(主图, log) + log 独有新增
        merged = dict(call_counts)
        n_new = 0
        n_existed = 0
        for k, v in log_call_counts.items():
            if k in merged:
                # 已存在的边, 取较大权重 (体现多源证据一致性)
                if v > merged[k]:
                    merged[k] = v
                n_existed += 1
            else:
                # 新增边
                merged[k] = v
                n_new += 1

        # 主图中标记"来自 log" 标签 (用于 r_trace 上限)
        from_log_keys = set(log_call_counts.keys())

        print(f"    [CausalGraphAgent] 步骤1.5: LogTopologyAgent 合并完成")
        print(f"    [CausalGraphAgent]   log 拓扑状态: {log_status}")
        print(f"    [CausalGraphAgent]   log 拓扑边数: {len(log_call_counts)}")
        print(f"    [CausalGraphAgent]   合并后调用对: {len(merged)} (新增 {n_new}, 共同 {n_existed})")
        call_counts = merged
        # 把"主图原始边"和"log 拓扑边"区分开 (传后续步骤)
        workspace["data_layer"]["_trace_call_counts"] = {
            f"{k[0]}->{k[1]}": v for k, v in trace_call_counts.items()
        }
        workspace["data_layer"]["_from_log_topology_keys"] = {
            f"{k[0]}->{k[1]}" for k in from_log_keys
        }

        # Step 4.3: 根据expand_mode选择图节点
        selected_nodes = select_graph_nodes(call_counts, refined, anomalous, expand_mode)
        if not selected_nodes:
            selected_nodes = set(refined or anomalous)
        # 把 log 拓扑独有的节点也加进来 (扩展图)
        for k in from_log_keys:
            selected_nodes.add(k[0])
            selected_nodes.add(k[1])
        print(f"    [CausalGraphAgent] 步骤2: 选中节点数={len(selected_nodes)}")

        graph = WeightedCausalGraph()
        anomaly_scores = workspace["association_layer"].get("anomaly_scores", {})
        intensity = workspace["association_layer"].get("component_intensity_series", {})
        first_ts = workspace["association_layer"].get("first_anomaly_ts", {})

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
        edge_scores = {}
        from_log_keys = workspace["data_layer"].get("_from_log_topology_keys", set())
        # log_topology 边在 call_counts 中也以 (caller, callee) 键存在 (来自合并 step 1.5)
        # 因此这里直接复用 call_counts, 但同时单独维护 log_call_counts_dict 用于
        # 查询某条边是否被 log_topology 确认, 与 trace 重合时给 r_trace 多源证据加成
        log_call_counts_dict = log_call_counts if isinstance(log_call_counts, dict) else {}
        for (caller, callee), count in call_counts.items():
            source = callee
            target = caller
            if source not in selected_nodes or target not in selected_nodes or source == target:
                continue

            # Scheme-F 四因子
            s_time = time_precedence_score(first_ts.get(source), first_ts.get(target), self.config)
            s_corr = lagged_corr_score(intensity.get(source), intensity.get(target))
            s_llm = self._domain_score(source, target, count, first_ts)
            s_prior = type_prior_score(source, target, self.config)

            # 融合权重（技术方案公式）
            if self.config.use_llm_edge_scoring and s_llm is not None:
                s_f = 0.25 * s_time + 0.25 * s_corr + 0.25 * s_llm + 0.25 * s_prior
            else:
                s_f = (
                    self.config.alpha_time * s_time
                    + self.config.alpha_corr * s_corr
                    + self.config.alpha_type_prior * s_prior
                )

            # 冲突调整（技术方案一致性检查）
            if self.config.use_llm_edge_scoring and s_llm is not None:
                if s_llm > 0.7 and s_time < 0.2:
                    s_f *= 0.6
                elif s_llm < 0.3 and s_corr > 0.8:
                    s_f = min(1.0, s_f * 1.3)

            trace_count = int(trace_call_counts.get((caller, callee), 0))
            is_inferred = trace_count <= 0
            is_from_log = f"{caller}->{callee}" in from_log_keys
            log_overlap_count = int(log_call_counts_dict.get((caller, callee), 0)) if is_from_log else 0
            if is_from_log and trace_count <= 0:
                # 来自 log 拓扑的边 (DBCP2 datasource / Jedis pool)
                # 中等置信度: 上限 0.55, 介于推断(0.45) 和 trace(1.0) 之间
                r_trace = min(0.55, 0.20 + 0.30 * max(s_time, s_corr))
                # 用一个较高的 prior_anchor, 因为 log 拓扑代表"业务配置层"已知关系
                prior_anchor = 0.50 * s_prior
                weight = r_trace * s_f + (1.0 - r_trace) * prior_anchor
            elif is_inferred:
                # Inferred dependency edges are structural hypotheses, not
                # observed traces. Let them help connect the graph, but make
                # their weight depend on telemetry support instead of type
                # prior alone.
                r_trace = min(0.45, 0.10 + 0.30 * max(s_time, s_corr))
                prior_anchor = 0.35 * s_prior
                weight = r_trace * s_f + (1.0 - r_trace) * prior_anchor
            else:
                r_trace = trace_reliability(trace_count, self.config)
                weight = r_trace * s_f + (1.0 - r_trace) * s_prior
            # 多源证据一致性加成: 若该 trace 边同时被 log_topology 确认 (access_log 等)
            # 视为独立证据交叉验证, 给予 r_trace 最多 +5% 上限
            if log_overlap_count > 0 and trace_count > 0:
                r_trace = min(1.0, r_trace + 0.05)
                weight = min(1.0, weight * 1.05)
            weight = max(0.0, min(1.0, weight))

            scores = {
                "observed_trace": trace_count > 0,
                "from_log_topology": is_from_log and trace_count <= 0,
                "confirmed_by_log": log_overlap_count > 0 and trace_count > 0,
            }
            graph.add_edge(source, target, weight=round(weight, 6), call_count=count, scores=scores)
            edge_detail = {
                "weight": round(weight, 6),
                "call_count": count,
                **scores,
            }
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
        # 优先路径: config.run_output_dir/causal_graphs/<dataset>/row_X/
        #   (这是 run_test.py 每次运行生成的时间戳目录)
        # 兜底路径: config.resolved_output_root()/<dataset>/row_X_causal_graphs/
        # ============================================================
        run_output_dir = getattr(self.config, "run_output_dir", None)
        if run_output_dir is not None:
            run_output_dir = Path(run_output_dir)
            case_dir = run_output_dir / "causal_graphs" / query.dataset / f"row_{query.row_id:04d}"
        else:
            output_root = Path(self.config.resolved_output_root()) if hasattr(self.config, "resolved_output_root") else None
            if output_root is not None:
                case_dir = output_root / query.dataset / f"row_{query.row_id}_causal_graphs"
            else:
                case_dir = None
        graph_save_dir: Path = None  # type: ignore
        if case_dir is not None:
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
        # ====================================================================
        # 图1: 调用图骨架 —— 打印
        # 数据来源: 全天 trace (raw_traces_full_day)
        # 构建方法: build_call_counts() —— 按 (trace_id, parent_id→span_id) 还原
        #           caller→callee，每对调用统计全天 call_count 作为权重
        # 方向语义: caller -> callee (调用方向)
        # ====================================================================
        print(f"    [CausalGraphAgent] --- 图1: 调用图骨架 (caller->callee, 权重=call_count) ---")
        print(f"      数据源 : 全天 trace  ({n_files_day} 个文件, 共 {n_rows_day} 个 span)")
        print(f"      时间窗 : 全天 24h（包含故障前后）")
        if tw:
            print(f"      故障窗 : [{tw.get('effective_start_ts', '?')}, {tw.get('end_ts', '?')}] (epoch秒) "
                  f"—— 仅用于图2相关性计算")
        print(f"      调用对 : 共 {len(call_counts)} 个 caller->callee 关系")
        sorted_call = sorted(call_counts.items(), key=lambda x: -x[1])
        for i, ((caller, callee), cnt) in enumerate(sorted_call, 1):
            print(f"      #{i:>2} {caller}({infer_component_type(caller)}) --(calls={cnt})--> {callee}({infer_component_type(callee)})")
        # ====================================================================
        # 图2: 因果图 —— 打印
        # 数据来源:
        #   - 节点: 来自图1调用图骨架的 selected_nodes (全天后筛选)
        #   - 边权重: Scheme-F 融合 = r_trace * s_F(故障窗口内span相关性)
        #                          + (1-r_trace) * s_prior(全局调用频次r_trace, 全天统计)
        # 方向语义: callee -> caller (故障传播方向，与图1相反)
        # 时间窗: 故障窗口内 (effective_start_ts, end_ts) 用于计算 s_F
        # ====================================================================
        # 图2: 因果图
        print(f"    [CausalGraphAgent] --- 图2: 因果图 (callee->caller, 故障传播方向, 权重=Scheme-F) ---")
        print(f"      数据源 : 节点←图1全天trace骨架, 边权重←Scheme-F(故障窗口span相关性+全天调用先验)")
        if tw:
            print(f"      时间窗 : 故障窗口 [{tw.get('effective_start_ts', '?')}, {tw.get('end_ts', '?')}] (epoch秒)")
            print(f"               (窗口 trace 文件数={n_files_win}, span行数={n_rows_win})")
        if graph.edges:
            for i, edge in enumerate(sorted(graph.edges, key=lambda x: -x.weight), 1):
                src_type = infer_component_type(edge.source)
                tgt_type = infer_component_type(edge.target)
                s = edge.scores or {}
                source_flag = "trace" if s.get("observed_trace") else "log" if s.get("from_log_topology") else "inferred"
                if s.get("confirmed_by_log"):
                    source_flag = "trace+log"
                print(f"      #{i:>2} {edge.source}({src_type}) --(w={edge.weight:.4f}, calls={edge.call_count}, source={source_flag})--> {edge.target}({tgt_type})")
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
