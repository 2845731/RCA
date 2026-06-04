from __future__ import annotations

import heapq
import math
from collections import defaultdict, deque
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

from causalrca_codex.config import AgentLoopConfig
from causalrca_codex.core.component import infer_component_type, normalize_component_id
from causalrca_codex.schemas import CallCounts, RCAQuery, WeightedCausalGraph


# ---------------------------------------------------------------------------
# 打印工具：醒目分隔线和状态输出
# ---------------------------------------------------------------------------
def _banner(msg: str, char: str = "=", width: int = 70) -> None:
    """打印醒目的分隔线标题。"""
    print(f"\n{char * width}")
    print(f"  {msg}")
    print(f"{char * width}\n")


def build_call_counts(trace_frames: Iterable[pd.DataFrame], query: RCAQuery) -> CallCounts:
    """从全天trace数据中提取调用关系计数。

    技术方案 Step 4.1：从全天trace中提取caller->callee调用图。
    调用识别规则：如果 span_B.parent_id = span_A.span_id，
    则 cmdb_id of span_A 调用了 cmdb_id of span_B。

    Args:
        trace_frames: 全天trace CSV文件的DataFrame列表
        query: RCA查询对象，包含候选组件列表用于ID归一化

    Returns:
        dict: {(caller, callee): call_count} 调用计数字典
    """
    call_counts: CallCounts = defaultdict(int)
    for df in trace_frames:
        # 兼容 TelemetryFrame: 取出 .data
        if hasattr(df, "data") and not isinstance(df, pd.DataFrame):
            df = df.data
        if isinstance(df, pd.DataFrame) and df.empty:
            continue
        if not isinstance(df, pd.DataFrame):
            continue
        if {"trace_id", "span_id", "parent_id", "cmdb_id"}.issubset(df.columns):
            _consume_trace(df, "trace_id", "span_id", "parent_id", "cmdb_id", query, call_counts)
        elif {"traceId", "id", "pid", "cmdb_id"}.issubset(df.columns):
            _consume_trace(df, "traceId", "id", "pid", "cmdb_id", query, call_counts)
        if {"cmdb_id", "dsName"}.issubset(df.columns):
            for _, row in df.dropna(subset=["cmdb_id", "dsName"]).iterrows():
                caller = normalize_component_id(row["cmdb_id"], query.candidate_components)
                callee = normalize_component_id(row["dsName"], query.candidate_components)
                if caller and callee and caller != callee:
                    call_counts[(caller, callee)] += 1
        if {"cmdb_id", "serviceName"}.issubset(df.columns):
            for _, row in df.dropna(subset=["cmdb_id", "serviceName"]).iterrows():
                caller = normalize_component_id(row["cmdb_id"], query.candidate_components)
                callee = normalize_component_id(row["serviceName"], query.candidate_components)
                if caller and callee and caller != callee:
                    call_counts[(caller, callee)] += 1
    return dict(call_counts)


def infer_missing_dependencies(
    call_counts: CallCounts,
    refined_candidates: List[str],
    query: RCAQuery,
) -> CallCounts:
    """⚠️ DEPRECATED: 此函数已被 CausalGraphAgent 禁用，引入噪声过大。

    推断 db/redis/node 组件的缺失依赖边。
    历史原因：在某些 trace 数据集里，数据库/缓存组件不出现在 span 中。
    推断规则（已弃用）:
      - db/redis -> service/pod: typical dependency (service calls db)
      - node -> pod/service: infrastructure dependency
      - 推断边 call_count = 5 (db) / 3 (node)

    问题:
      1) n×m 全连接：对每个 db 配对所有 service，n×m 条臆测边
      2) 完全不知道 db 实际被哪些 service 调用，一律假定全部
      3) 让所有 db 看起来都"差不多像根因"，稀释真实异常路径

    替代方案: 让 expand_mode="full_path" 在 BFS 时基于真实 trace 边自然
              补全节点；如确需补全边，应从 cmdb_dependency.yaml 配置文件
              读取，而不是从组件名字符串推断。

    本函数保留在此处仅供"配置驱动版本"未来参考。
    """
    import warnings
    warnings.warn(
        "infer_missing_dependencies 已弃用, 会在 CausalGraphAgent 中引入噪声。"
        "请使用基于 cmdb_dependency.yaml 的配置驱动方案。",
        DeprecationWarning,
        stacklevel=2,
    )
    inferred: CallCounts = defaultdict(int)

    # Separate candidates by type
    db_redis = []
    services = []
    nodes = []
    others = []
    for comp in refined_candidates:
        comp_type = infer_component_type(comp)
        if comp_type in ("db", "redis"):
            db_redis.append(comp)
        elif comp_type in ("service", "pod"):
            services.append(comp)
        elif comp_type == "node":
            nodes.append(comp)
        else:
            others.append(comp)

    # If db/redis components exist but have no trace edges, add inferred edges to services
    for db_comp in db_redis:
        has_trace_edge = any(db_comp in pair for pair in call_counts)
        if not has_trace_edge:
            # Infer: services depend on db/redis (caller -> callee)
            # In causal graph, this becomes db/redis -> services (reversed)
            for svc in services:
                inferred[(svc, db_comp)] = 5  # Low confidence inferred edge

    # If node components exist but have no trace edges, add inferred edges
    for node_comp in nodes:
        has_trace_edge = any(node_comp in pair for pair in call_counts)
        if not has_trace_edge:
            for svc in services + db_redis:
                inferred[(svc, node_comp)] = 3  # Even lower confidence

    if inferred:
        # Merge with existing call_counts
        merged = dict(call_counts)
        for pair, count in inferred.items():
            merged[pair] = merged.get(pair, 0) + count
        return merged
    return call_counts


def _consume_trace(
    df: pd.DataFrame,
    trace_col: str,
    span_col: str,
    parent_col: str,
    comp_col: str,
    query: RCAQuery,
    call_counts: CallCounts,
) -> None:
    """从 trace DataFrame 中按 (caller, callee) 累计调用次数 (向量化版).

    约定 (与旧版一致):
      parent_span -> child_span 表示 parent 调用 child.
      即 caller = parent 的 component, callee = child 的 component.
    性能: 12M 行从 ~20-30 分钟降至 < 1 分钟.
    """
    use = df[[trace_col, span_col, parent_col, comp_col]].dropna(subset=[trace_col, span_col, comp_col])
    n_total = len(use)
    n_traces = use[trace_col].nunique()
    if n_total > 100_000:
        print(f"    [graph_ops] _consume_trace 进度: 总行数={n_total}, trace 数={n_traces} (向量化)", flush=True)

    # 1) (trace_id, span_id) -> normalized_component
    span_norm = (
        use.groupby([trace_col, span_col])[comp_col]
        .first()
        .map(lambda c: normalize_component_id(c, query.candidate_components))
    )
    # 2) 取出每行: child_span 的 component = callee
    #    parent 的 component    = caller
    span_norm_df = span_norm.reset_index()
    span_norm_df.columns = [trace_col, "_span_key", "_component"]
    use2 = use[[trace_col, span_col, parent_col]].copy()
    use2["_callee_key"] = use2[span_col]   # child span
    use2["_caller_key"] = use2[parent_col] # parent span
    # merge callee: child_span -> component
    callee_lookup = span_norm_df.rename(columns={"_span_key": "_callee_key", "_component": "callee_norm"})
    use2 = use2.merge(
        callee_lookup[[trace_col, "_callee_key", "callee_norm"]],
        on=[trace_col, "_callee_key"],
        how="left",
    )
    # merge caller: parent_span -> component
    caller_lookup = span_norm_df.rename(columns={"_span_key": "_caller_key", "_component": "caller_norm"})
    use2 = use2.merge(
        caller_lookup[[trace_col, "_caller_key", "caller_norm"]],
        on=[trace_col, "_caller_key"],
        how="left",
    )
    # 3) 过滤并累加
    pair = use2[["caller_norm", "callee_norm"]].dropna()
    pair = pair[(pair["caller_norm"] != "") & (pair["callee_norm"] != "")]
    pair = pair[pair["caller_norm"] != pair["callee_norm"]]
    if not pair.empty:
        vc = pair.value_counts()
        for (caller, callee), cnt in vc.items():
            if caller and callee:
                call_counts[(caller, callee)] += int(cnt)
    if n_total > 100_000:
        print(f"    [graph_ops] _consume_trace 完成: call_counts={len(call_counts)} 对", flush=True)


def select_graph_nodes(
    call_counts: CallCounts,
    refined_candidates: Iterable[str],
    anomalous_components: Iterable[str],
    expand_mode: str,
) -> Set[str]:
    """根据expand_mode选择因果图的节点集合。

    技术方案 Step 4.3 图扩展：
    - "none": 仅包含精炼候选组件 C_refined
    - "direct": 加入直接调用邻居（含正常组件），用于图无边时
    - "full_path": BFS找候选组件间最短路径上的所有节点

    Args:
        call_counts: 调用计数字典
        refined_candidates: 精炼后的候选组件列表
        anomalous_components: 所有异常组件列表
        expand_mode: 扩展模式 "none"/"direct"/"full_path"

    Returns:
        选中的节点集合
    """
    refined = set(refined_candidates)
    anomalous = set(anomalous_components)
    nodes = set(refined or anomalous)
    if expand_mode == "none":
        return nodes
    neighbors = set(nodes)
    for caller, callee in call_counts:
        if caller in nodes or callee in nodes:
            neighbors.update([caller, callee])
    if expand_mode == "direct":
        return neighbors
    if expand_mode == "full_path":
        graph = defaultdict(set)
        undirected = defaultdict(set)
        for caller, callee in call_counts:
            graph[caller].add(callee)
            undirected[caller].add(callee)
            undirected[callee].add(caller)
        expanded = set(neighbors)
        targets = list(nodes)
        for source in targets:
            parents = {source: None}
            queue = deque([source])
            while queue:
                current = queue.popleft()
                for nxt in undirected.get(current, set()):
                    if nxt not in parents:
                        parents[nxt] = current
                        queue.append(nxt)
            for target in targets:
                if target not in parents:
                    continue
                cur = target
                while cur is not None:
                    expanded.add(cur)
                    cur = parents[cur]
        return expanded
    return nodes


def lagged_corr_score(series_x: Optional[pd.DataFrame], series_y: Optional[pd.DataFrame], max_lag: int = 5) -> float:
    """Scheme-B: 计算两个KPI时间序列的滞后Pearson相关性。

    技术方案要求：
    - lag搜索范围 0~5 步（每步1分钟）
    - 使用相关系数的绝对值（传播可以是正相关或负相关）

    Args:
        series_x: X组件的最异常KPI时间序列 DataFrame(timestamp, intensity)
        series_y: Y组件的最异常KPI时间序列 DataFrame(timestamp, intensity)
        max_lag: 最大滞后步数，默认5

    Returns:
        0~1 之间的相关性得分（取绝对值后的最大值）
        注意：无法计算时返回 0.00001（而非 0.0），避免融合时直接抹杀证据
    """
    if series_x is None or series_y is None or series_x.empty or series_y.empty:
        return 0.00001
    merged = pd.merge(
        series_x.rename(columns={"intensity": "x"}),
        series_y.rename(columns={"intensity": "y"}),
        on="timestamp",
        how="outer",
    ).fillna(0.0)
    if len(merged) < 3:
        return 0.00001
    best = 0.0
    for lag in range(max_lag + 1):
        shifted = merged["y"].shift(-lag)
        corr = merged["x"].corr(shifted)
        if corr == corr:  # 排除 NaN
            best = max(best, abs(float(corr)))  # 取绝对值：传播可正可负
    return max(0.00001, min(1.0, best))


def time_precedence_score(t_x: Optional[int], t_y: Optional[int], config: AgentLoopConfig) -> float:
    """Scheme-A: 时间序证据得分。

    技术方案公式：
    - dt > 0（Y在X之后异常）: score = 1/(1 + dt/60)，dt单位为秒
    - dt = 0（同时异常）: score = 0.5
    - dt < 0（Y在X之前异常）: score = 0.1

    注意：无法计算时返回 0.00001（而非 0.5），避免融合时时间证据不起作用

    Args:
        t_x: X组件首次异常时间（epoch秒）
        t_y: Y组件首次异常时间（epoch秒）
        config: 配置对象（保留用于未来扩展）

    Returns:
        0~1 之间的时间序得分
    """
    if t_x is None or t_y is None:
        return 0.00001
    dt = t_y - t_x  # 秒
    if dt > 0:
        return 1.0 / (1.0 + dt / 60.0)
    if dt == 0:
        return 0.00001
    return 0.1


def type_prior_score(x: str, y: str, config: AgentLoopConfig) -> float:
    """Scheme-D: 组件类型先验得分。

    技术方案：基于SRE经验的确定性先验概率表。
    例如：host/node -> pod = 0.90, db -> pod/service = 0.80, pod -> pod = 0.60

    Args:
        x: 源组件名（原因端）
        y: 目标组件名（结果端）
        config: 配置对象，包含 type_prior_table

    Returns:
        先验概率得分
    """
    source_type = infer_component_type(x)
    target_type = infer_component_type(y)
    return config.type_prior_table.get((source_type, target_type), 0.45)


def heuristic_domain_score(x: str, y: str, edge_is_trace_backed: bool) -> float:
    """Domain knowledge score based on PRIOR table from technical solution."""
    x_type = infer_component_type(x)
    y_type = infer_component_type(y)
    # Match PRIOR table values from technical solution Section 11.1.4
    prior = {
        ("node", "pod"): 0.90, ("node", "service"): 0.80,
        ("node", "db"): 0.85, ("node", "redis"): 0.85,
        ("db", "pod"): 0.80, ("db", "service"): 0.80,
        ("redis", "pod"): 0.75, ("redis", "service"): 0.75,
        ("pod", "pod"): 0.60, ("service", "service"): 0.60,
        ("pod", "service"): 0.60, ("service", "pod"): 0.60,
    }
    score = prior.get((x_type, y_type), 0.50 if edge_is_trace_backed else 0.35)
    return score


def trace_reliability(call_count: int, config: AgentLoopConfig) -> float:
    return 1.0 - math.exp(-float(call_count) / max(config.lambda_call_count, 1e-6))


def max_path_strengths(graph: WeightedCausalGraph, source: str, max_depth: int = 6) -> Dict[str, float]:
    """计算瓶颈模型的InfluenceScore（技术方案 Agent 5 核心算法）。

    技术方案要求使用 max-min 瓶颈路径模型，而非路径权重乘积模型：
        InfScore(X->Y) = max over all paths p from X->Y of [ min over edges in p of w(e) ]

    这是因为故障传播存在放大效应（重试风暴、级联故障），乘积模型会随路径长度
    单调衰减，而瓶颈模型保留了最弱环节语义。

    使用修改版 Dijkstra 算法（最大堆），复杂度 O(|E| log |V|)。

    Args:
        graph: 带权因果图
        source: 源节点
        max_depth: 最大搜索深度

    Returns:
        dict: {target_node: influence_score}，source自身的InfluenceScore为1.0
    """
    # 最大堆：(-score, node, depth)，Python heapq是最小堆，取负实现最大堆
    best: Dict[str, float] = {source: 1.0}
    heap: List[Tuple[float, str, int]] = [(-1.0, source, 0)]

    while heap:
        neg_score, node, depth = heapq.heappop(heap)
        current_score = -neg_score

        # 如果已找到更优路径，跳过
        if current_score < best.get(node, 0.0) - 1e-12:
            continue

        if depth >= max_depth:
            continue

        for edge in graph.outgoing(node):
            # 瓶颈模型：取路径上的最小边权
            bottleneck = min(current_score, edge.weight)
            if bottleneck > best.get(edge.target, 0.0) + 1e-12:
                best[edge.target] = bottleneck
                heapq.heappush(heap, (-bottleneck, edge.target, depth + 1))

    return best
