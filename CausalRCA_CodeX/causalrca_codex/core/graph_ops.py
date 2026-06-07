from __future__ import annotations

import heapq
import math
from collections import defaultdict, deque
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
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
    """Infer missing dependency edges for db/redis/node components.

    In many trace datasets, database and Redis components don't appear as cmdb_id
    in trace spans. However, service components (Tomcat, MG, IG) typically depend on
    databases and caches. This function adds inferred edges based on domain knowledge:
    - db/redis -> service/pod: typical dependency (service calls db)
    - node -> pod/service: infrastructure dependency

    These inferred edges get lower call counts (5) to reflect lower confidence.
    """
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
    use = df[[trace_col, span_col, parent_col, comp_col]].dropna(subset=[trace_col, span_col, comp_col])
    for _, group in use.groupby(trace_col):
        span_to_component = {
            str(row[span_col]): normalize_component_id(row[comp_col], query.candidate_components)
            for _, row in group.iterrows()
        }
        for _, row in group.iterrows():
            parent = str(row[parent_col])
            child_span = str(row[span_col])
            if parent in span_to_component and child_span in span_to_component:
                caller = span_to_component[parent]
                callee = span_to_component[child_span]
                if caller and callee and caller != callee:
                    call_counts[(caller, callee)] += 1


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
    """
    if series_x is None or series_y is None or series_x.empty or series_y.empty:
        return 0.0
    merged = pd.merge(
        series_x.rename(columns={"intensity": "x"}),
        series_y.rename(columns={"intensity": "y"}),
        on="timestamp",
        how="outer",
    ).fillna(0.0)
    if len(merged) < 3:
        return 0.0
    best = 0.0
    for lag in range(max_lag + 1):
        shifted = merged["y"].shift(-lag)
        corr = merged["x"].corr(shifted)
        if corr == corr:  # 排除 NaN
            best = max(best, abs(float(corr)))  # 取绝对值：传播可正可负
    return max(0.0, min(1.0, best))


def granger_causality_score(
    series_x: Optional[pd.DataFrame],
    series_y: Optional[pd.DataFrame],
    max_lag: int = 5,
) -> float:
    """Innovation: Granger-style causality score for directional causal evidence.

    Unlike symmetric correlation (Scheme-B), this tests DIRECTIONAL precedence:
    if X's anomalies help predict Y's anomalies better than Y predicts X,
    that's evidence for X→Y causality (X is the cause, Y is the effect).

    Algorithm:
    1. Compute cross-correlation at positive lags (X leads Y) and negative lags (Y leads X)
    2. best_forward = max |corr| at lag 1..max_lag (X leads Y)
    3. best_reverse = max |corr| at lag -1..-max_lag (Y leads X)
    4. directional_score = (best_forward - best_reverse + 1) / 2
       - If X strongly leads Y: best_forward >> best_reverse → score ≈ 1.0
       - If Y strongly leads X: best_forward << best_reverse → score ≈ 0.0
       - If symmetric (no direction): score ≈ 0.5

    This is a proper Granger-inspired directional signal that complements
    the symmetric lagged correlation (Scheme-B).

    Args:
        series_x: X component's anomaly intensity series (source in causal graph)
        series_y: Y component's anomaly intensity series (target in causal graph)
        max_lag: Maximum lag steps (default 5)

    Returns:
        0~1 score: higher = more evidence that X→Y (X causes Y)
    """
    if series_x is None or series_y is None or series_x.empty or series_y.empty:
        return 0.5  # No data → neutral

    merged = pd.merge(
        series_x.rename(columns={"intensity": "x"}),
        series_y.rename(columns={"intensity": "y"}),
        on="timestamp",
        how="outer",
    ).fillna(0.0)
    if len(merged) < 4:
        return 0.5

    # Forward: X leads Y (positive lag = Y is shifted backward)
    best_forward = 0.0
    for lag in range(1, max_lag + 1):
        shifted_y = merged["y"].shift(-lag)
        corr = merged["x"].corr(shifted_y)
        if corr == corr:  # not NaN
            best_forward = max(best_forward, abs(float(corr)))

    # Reverse: Y leads X (negative lag = X is shifted backward)
    best_reverse = 0.0
    for lag in range(1, max_lag + 1):
        shifted_x = merged["x"].shift(-lag)
        corr = merged["y"].corr(shifted_x)
        if corr == corr:
            best_reverse = max(best_reverse, abs(float(corr)))

    # Directional score: normalized difference
    # If forward >> reverse: score near 1.0 (X causes Y)
    # If forward << reverse: score near 0.0 (Y causes X)
    # If symmetric: score near 0.5
    total = best_forward + best_reverse
    if total < 1e-6:
        return 0.5
    directional = (best_forward - best_reverse + total) / (2.0 * total)
    return max(0.0, min(1.0, directional))


# Module-level cache for merged DataFrames
_MERGE_CACHE: Dict[str, pd.DataFrame] = {}


def combined_corr_scores(
    series_x: Optional[pd.DataFrame],
    series_y: Optional[pd.DataFrame],
    max_lag: int = 5,
) -> Tuple[float, float]:
    """Compute both lagged_corr_score and granger_causality_score in one pass.

    This avoids redundant pd.merge() and correlation computations.
    Returns (lagged_corr, granger_causality) tuple.
    """
    if series_x is None or series_y is None or series_x.empty or series_y.empty:
        return 0.0, 0.5

    # Cache key based on series identity (use id for speed)
    cache_key = f"{id(series_x)}:{id(series_y)}:{max_lag}"
    if cache_key in _MERGE_CACHE:
        merged = _MERGE_CACHE[cache_key]
    else:
        merged = pd.merge(
            series_x.rename(columns={"intensity": "x"}),
            series_y.rename(columns={"intensity": "y"}),
            on="timestamp",
            how="outer",
        ).fillna(0.0)
        if len(merged) >= 3:
            _MERGE_CACHE[cache_key] = merged

    n = len(merged)
    if n < 3:
        return 0.0, 0.5

    x_vals = merged["x"].values.astype(np.float64)
    y_vals = merged["y"].values.astype(np.float64)

    # Vectorized correlation computation using numpy
    def _corr_at_lag(arr_x: np.ndarray, arr_y: np.ndarray, lag: int) -> float:
        if lag == 0:
            a, b = arr_x, arr_y
        elif lag > 0:
            a, b = arr_x[:len(arr_x)-lag], arr_y[lag:]
        else:
            a, b = arr_x[-lag:], arr_y[:len(arr_y)+lag]
        if len(a) < 3:
            return float("nan")
        # Pearson correlation via numpy (faster than pandas)
        a_mean = np.mean(a)
        b_mean = np.mean(b)
        a_centered = a - a_mean
        b_centered = b - b_mean
        numerator = np.sum(a_centered * b_centered)
        denominator = np.sqrt(np.sum(a_centered**2) * np.sum(b_centered**2))
        if denominator < 1e-12:
            return float("nan")
        return float(numerator / denominator)

    # lagged_corr_score: max |corr| at lag 0..max_lag
    best_lagged = 0.0
    for lag in range(max_lag + 1):
        c = _corr_at_lag(x_vals, y_vals, lag)
        if c == c:  # not NaN
            best_lagged = max(best_lagged, abs(c))

    # granger_causality_score: forward vs reverse
    best_forward = 0.0
    for lag in range(1, max_lag + 1):
        c = _corr_at_lag(x_vals, y_vals, lag)
        if c == c:
            best_forward = max(best_forward, abs(c))

    best_reverse = 0.0
    for lag in range(1, max_lag + 1):
        c = _corr_at_lag(y_vals, x_vals, lag)
        if c == c:
            best_reverse = max(best_reverse, abs(c))

    total = best_forward + best_reverse
    if total < 1e-6:
        granger = 0.5
    else:
        granger = (best_forward - best_reverse + total) / (2.0 * total)
        granger = max(0.0, min(1.0, granger))

    return max(0.0, min(1.0, best_lagged)), granger


def time_precedence_score(t_x: Optional[int], t_y: Optional[int], config: AgentLoopConfig) -> float:
    """Scheme-A: 时间序证据得分。

    技术方案公式：
    - dt > 0（Y在X之后异常）: score = 1/(1 + dt/60)，dt单位为秒
    - dt = 0（同时异常）: score = 0.5
    - dt < 0（Y在X之前异常）: score = 0.1

    Args:
        t_x: X组件首次异常时间（epoch秒）
        t_y: Y组件首次异常时间（epoch秒）
        config: 配置对象（保留用于未来扩展）

    Returns:
        0~1 之间的时间序得分
    """
    if t_x is None or t_y is None:
        return 0.5
    dt = t_y - t_x  # 秒
    if dt > 0:
        return 1.0 / (1.0 + dt / 60.0)
    if dt == 0:
        return 0.5
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


def causal_topology_score(
    graph: WeightedCausalGraph,
    component: str,
    anomaly_scores: Dict[str, float],
) -> float:
    """计算因果拓扑评分（Causal Topology Score, CTS）。

    基于因果图的图论结构性质，量化节点"像根因"的程度。
    根因特征：高出度、低入度、能到达更多异常节点。

    CTS(X) = α · reach_ratio + β · degree_ratio - γ · anomalous_influence

    其中：
    - reach_ratio = |R(X)| / |A|：X 能到达的异常节点比例
    - degree_ratio = out / (out + in)：出度占比
    - anomalous_influence = incoming_from_anomalous / total_weight：被异常父节点影响的程度

    这是一个图论结构性质，不依赖于特定数据集的统计规律。

    Args:
        graph: 带权因果图
        component: 待评估的组件
        anomaly_scores: 所有异常组件的严重度分数

    Returns:
        0~1 之间的 CTS 分数，越高越像根因
    """
    anomalous_set = set(anomaly_scores.keys())
    if not anomalous_set or not graph.has_node(component):
        return 0.5

    # 1. Reach ratio: BFS from component, count reachable anomalous nodes
    reachable = set()
    queue = [component]
    visited = {component}
    while queue:
        node = queue.pop(0)
        for edge in graph.outgoing(node):
            if edge.target not in visited:
                visited.add(edge.target)
                if edge.target in anomalous_set:
                    reachable.add(edge.target)
                queue.append(edge.target)
    reach_ratio = len(reachable) / len(anomalous_set) if anomalous_set else 0.0

    # 2. Degree ratio: out / (out + in)
    out_degree = len(list(graph.outgoing(component)))
    in_degree = len(list(graph.incoming(component)))
    degree_ratio = out_degree / (out_degree + in_degree + 1e-9)

    # 3. Anomalous influence: incoming weight from anomalous parents / total weight
    incoming_from_anomalous = sum(
        float(getattr(e, 'weight', 0.5))
        for e in graph.incoming(component)
        if e.source in anomaly_scores
    )
    total_incoming = sum(float(getattr(e, 'weight', 0.5)) for e in graph.incoming(component))
    total_outgoing = sum(float(getattr(e, 'weight', 0.5)) for e in graph.outgoing(component))
    total_weight = total_incoming + total_outgoing + 1e-9
    anomalous_influence = incoming_from_anomalous / total_weight

    cts = 0.4 * reach_ratio + 0.35 * degree_ratio - 0.25 * anomalous_influence
    return max(0.0, min(1.0, cts))
