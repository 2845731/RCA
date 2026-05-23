from __future__ import annotations

from typing import Any, Dict, List


SPECTRAL_METRIC_SYSTEM = """你是一个云原生系统的频域异常分析专家。你的任务是根据指标的频域特征分析结果，解释异常的含义和可能的根因。

你需要用简洁的中文回答，重点关注：
1. 这个频域异常类型意味着什么物理现象
2. 这种异常通常由什么类型的故障引起
3. 这个异常与根因定位的关系

频域异常类型说明：
- slow_trend_anomaly（慢趋势异常）：低频能量集中，表示指标在缓慢持续偏移，通常对应资源耗尽（内存泄漏、磁盘满等）
- fast_burst_or_jitter_anomaly（突发抖动异常）：高频能量集中，表示指标突然剧烈波动，通常对应突发故障（网络抖动、进程崩溃等）
- periodic_oscillation_anomaly（周期振荡异常）：存在主导频率，表示指标周期性波动，通常对应定时任务干扰或配置错误
- mixed_spectral_anomaly（混合频域异常）：多个频段同时异常，表示复杂故障模式
- no_strong_spectral_anomaly（无明显频域异常）：频域特征不显著，可能只是轻微波动
"""

SPECTRAL_METRIC_USER = """请分析以下指标的频域异常检测结果：

指标名称: {node_id}
传统异常分数: {traditional_score:.4f}
频域异常分数: {spectral_score:.4f}
融合异常分数: {final_score:.4f}
频域异常类型: {anomaly_type}
频域特征:
  - 总能量: {total_energy:.4f}
  - 低频段能量比: {low_ratio:.4f}
  - 中频段能量比: {mid_ratio:.4f}
  - 高频段能量比: {high_ratio:.4f}
  - 主导频率能量比: {dominant_ratio:.4f}
  - 频谱熵: {spectral_entropy:.4f}
  - 最大稳健z分数: {max_z:.4f}
  - 最大变化率: {max_change_rate:.4f}

请用2-3句话解释：(1)这个频域类型意味着什么 (2)最可能的故障原因 (3)对根因定位的提示"""

LOG_ANALYSIS_SYSTEM = """你是一个云原生系统的日志分析专家。你的任务是根据日志中的错误模式，总结故障特征和可能的根因。

你需要用简洁的中文回答，重点关注：
1. 错误日志的模式和特征
2. 这些错误通常关联什么故障
3. 对根因定位的建议

常见错误关键词含义：
- error/exception/fail: 通用错误，需结合上下文判断
- timeout: 响应超时，可能是下游服务慢或网络问题
- refused/connection reset: 连接被拒绝，可能是服务不可用
- out of memory/oom: 内存不足，可能是内存泄漏
- deadlock: 死锁，可能是并发问题
- overflow: 溢出，可能是资源超限
"""

LOG_ANALYSIS_USER = """请分析以下组件的日志错误模式：

组件: {component}
错误日志数量: {error_count}
总日志数量: {total_logs}
错误比例: {error_ratio:.4f}
错误类型分布: {error_types}
异常分数: {anomaly_score:.4f}

请用2-3句话总结：(1)这个组件的日志错误模式 (2)最可能的故障原因 (3)是否可能是根因"""

CAUSAL_SYNTHESIS_SYSTEM = """你是一个云原生系统的因果推理专家。你的任务是根据异常证据和因果图分析结果，解释根因推理的逻辑链路。

你需要用简洁的中文回答，重点关注：
1. 为什么排在前面的节点更可能是根因
2. 因果传播路径是怎样的
3. 频域证据如何支持这个结论

根因排序依据：
- 异常分数越高，该节点越异常
- 出边权重越大，该节点越可能是传播源头（根因）
- 入边权重越大，该节点越可能是被影响的下游
- 频域异常类型提供故障模式的语义信息
"""

CAUSAL_SYNTHESIS_USER = """请解释以下根因排序结果：

故障时间窗口: {start_time} ~ {end_time}

Top-5 根因候选:
{top5_text}

因果图统计:
  - 候选边数: {total_edges}
  - 保留边数: {kept_edges}
  - 频域边验证保留率: {keep_rate:.1%}

请用3-4句话解释：(1)为什么Top-1是最可能的根因 (2)因果传播路径 (3)频域证据如何支持这个结论"""

REFLECTOR_SYSTEM = """你是一个云原生系统诊断质量评估专家。你的任务是根据多个Agent的诊断结果，评估诊断的可靠性，并给出改进建议。

你需要用简洁的中文回答，重点关注：
1. 当前诊断是否可靠（证据是否充分、多源是否一致）
2. 是否需要回溯重新假设
3. 如果需要改进，应该从哪个方向入手

评估标准：
- 多个Agent（指标/日志/Trace）是否指向同一个组件
- 置信度是否足够高（>0.5为可信）
- 因果图是否有足够的传播路径支撑
"""

REFLECTOR_USER = """请评估以下诊断结果的质量：

最大置信度: {max_confidence:.4f}
被支持的节点数: {supported_nodes}
各Agent的证据类型: {evidence_types}
因果分析Top-1: {top1_text}
因果分析Top-1分数: {top1_score:.4f}

请用2-3句话评估：(1)诊断是否可靠 (2)是否需要回溯 (3)改进建议"""

TRACE_ANALYSIS_SYSTEM = """你是一个云原生系统的调用链分析专家。你的任务是根据Trace数据中的调用链和延迟信息，分析故障传播路径。

你需要用简洁的中文回答，重点关注：
1. 哪个服务的延迟异常最显著
2. 调用链上的延迟传播方向
3. 这对根因定位的提示

分析原则：
- 延迟异常最严重的服务不一定是根因，它可能是被下游拖慢的
- 根因通常是调用链中最早出现异常的服务
- 需要结合频域分析判断异常是"自身问题"还是"被拖慢"
"""

TRACE_ANALYSIS_USER = """请分析以下调用链的延迟异常：

异常Span列表:
{anomalous_spans_text}

调用链结构:
{call_chains_text}

请用2-3句话分析：(1)哪个服务延迟最异常 (2)延迟传播方向 (3)对根因的判断"""


def format_spectral_metric_prompt(evidence: Dict[str, Any]) -> str:
    return SPECTRAL_METRIC_USER.format(
        node_id=evidence.get("node_id", "unknown"),
        traditional_score=evidence.get("traditional_score", 0.0),
        spectral_score=evidence.get("spectral_score", 0.0),
        final_score=evidence.get("final_anomaly_score", 0.0),
        anomaly_type=evidence.get("anomaly_type", "unknown"),
        total_energy=evidence.get("total_energy", 0.0),
        low_ratio=evidence.get("low_ratio", 0.0),
        mid_ratio=evidence.get("mid_ratio", 0.0),
        high_ratio=evidence.get("high_ratio", 0.0),
        dominant_ratio=evidence.get("dominant_freq_energy_ratio", 0.0),
        spectral_entropy=evidence.get("spectral_entropy", 0.0),
        max_z=evidence.get("max_abs_robust_z", 0.0),
        max_change_rate=evidence.get("max_change_rate", 0.0),
    )


def format_log_prompt(component: str, error_info: Dict[str, Any]) -> str:
    return LOG_ANALYSIS_USER.format(
        component=component,
        error_count=error_info.get("error_count", 0),
        total_logs=error_info.get("total_logs", 0),
        error_ratio=error_info.get("error_ratio", 0.0),
        error_types=error_info.get("error_types", {}),
        anomaly_score=error_info.get("anomaly_score", 0.0),
    )


def format_causal_prompt(
    top5: List[Dict[str, Any]],
    total_edges: int,
    kept_edges: int,
    start_time: str,
    end_time: str,
) -> str:
    top5_lines = []
    for i, c in enumerate(top5):
        top5_lines.append(
            f"  {i+1}. {c.get('node_id', 'unknown')}: "
            f"根因分数={c.get('root_score', 0.0):.4f}, "
            f"异常分数={c.get('anomaly_score', 0.0):.4f}, "
            f"出边证据={c.get('out_evidence', 0.0):.4f}"
        )
    top5_text = "\n".join(top5_lines)
    keep_rate = kept_edges / max(total_edges, 1)
    return CAUSAL_SYNTHESIS_USER.format(
        top5_text=top5_text,
        total_edges=total_edges,
        kept_edges=kept_edges,
        keep_rate=keep_rate,
        start_time=start_time,
        end_time=end_time,
    )


def format_reflector_prompt(
    max_confidence: float,
    supported_nodes: int,
    evidence_types: List[str],
    top1_text: str,
    top1_score: float,
) -> str:
    return REFLECTOR_USER.format(
        max_confidence=max_confidence,
        supported_nodes=supported_nodes,
        evidence_types=", ".join(evidence_types),
        top1_text=top1_text,
        top1_score=top1_score,
    )


def format_trace_prompt(
    anomalous_spans: List[Dict[str, Any]],
    call_chains: List[Dict[str, Any]],
) -> str:
    span_lines = []
    for s in anomalous_spans[:10]:
        span_lines.append(
            f"  - {s.get('cmdb_id', 'unknown')}: "
            f"中位延迟={s.get('median_duration', 0):.1f}ms, "
            f"z分数={s.get('z_score', 0):.2f}, "
            f"异常分数={s.get('anomaly_score', 0):.4f}"
        )
    spans_text = "\n".join(span_lines) if span_lines else "  无异常Span"

    chain_lines = []
    for c in call_chains[:5]:
        chain_lines.append(
            f"  - 调用链 {c.get('trace_id', '?')}: "
            f"{' -> '.join(c.get('components', []))}"
        )
    chains_text = "\n".join(chain_lines) if chain_lines else "  无调用链数据"

    return TRACE_ANALYSIS_USER.format(
        anomalous_spans_text=spans_text,
        call_chains_text=chains_text,
    )
