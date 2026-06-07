from __future__ import annotations

from typing import Any, Dict, List, Tuple

from causalrca_codex.agents.base import BaseAgent
from causalrca_codex.core.telemetry import iter_metric_series, load_day_frames, window_frame
from causalrca_codex.schemas import RCAQuery


class DataAgent(BaseAgent):
    """Agent 1: 数据加载与预处理Agent（技术方案 Step 1）。

    职责：
    1. 加载全天metric CSV文件
    2. KPI聚合：按 (cmdb_id, kpi_name) 分组
    3. 全局阈值计算：基于全天数据的P95/P5百分位数
    4. 故障窗口过滤：仅保留 [t_s, t_e] 内的数据
    5. 加载全天trace和窗口日志

    数据范围规则（技术方案关键设计）：
    - 阈值计算：全天数据（用故障窗口会稀释异常）
    - 异常检测：故障窗口
    - 调用图：全天trace（窗口trace可能不完整）
    - 日志分析：故障窗口
    """

    name = "DataAgent"
    purpose = "加载遥测数据 + 全天阈值计算 + 故障窗口过滤"
    preconditions = ["workspace.task.query"]
    produces = ["data_layer.component_kpi_series", "data_layer.global_thresholds", "data_layer.raw_traces", "data_layer.raw_logs"]
    tunable_params = {"threshold_percentile": 95.0, "aggregate": "mean"}

    def _execute(self, workspace: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        # ============================================================
        # 步骤目的: 加载遥测数据 + 全天阈值计算 + 故障窗口过滤
        # 计算方法: ①读 metric.csv(全天)按 component+kpi 聚合
        #          ②计算全局上下阈值 P95 / P5
        #          ③过滤出 query.start_time~end_time 内的 trace/log
        # 读取数据: dataset/{system}/metric.csv(全量指标)
        #          dataset/{system}/trace.csv(全量调用)
        #          dataset/{system}/log.csv(全量日志)
        # ============================================================
        query: RCAQuery = workspace["task"]["query"]
        threshold_percentile = float(params.get("threshold_percentile", self.config.threshold_percentile))
        low_percentile = float(params.get("low_percentile", self.config.low_percentile))
        aggregate = str(params.get("aggregate", "mean"))

        # RootCandidateRecovery: expand metric window backward
        expand_minutes = int(params.get("window_expand_minutes", 0))
        effective_start_ts = query.start_ts - expand_minutes * 60

        # Step 1.1: 加载全天metric文件
        print(f"    [DataAgent] 步骤1: 加载 dataset/{query.dataset}/metric.csv + trace.csv + log.csv")
        frames = load_day_frames(self.config, query)

        # Step 1.2 + 1.3: KPI聚合 + 全局阈值计算
        metric_series = iter_metric_series(
            frames["metric"],
            query,
            aggregate=aggregate,
            threshold_percentile=threshold_percentile,
            low_percentile=low_percentile,
            window_start_ts=effective_start_ts,
        )

        # Step 1.4: 故障窗口过滤 (expanded)
        trace_windows = [window_frame(frame, effective_start_ts, query.end_ts) for frame in frames["trace"]]
        log_windows = [window_frame(frame, effective_start_ts, query.end_ts) for frame in frames["log"]]

        component_kpi_series = {}
        thresholds = {}
        for series in metric_series:
            key = f"{series.component}|{series.kpi}|{series.file_name}"
            component_kpi_series[key] = series
            thresholds[key] = {
                "high": series.threshold_high,
                "low": series.threshold_low,
                "median": series.median,
                "scale": series.scale,
                "threshold_percentile": threshold_percentile,
                "low_percentile": low_percentile,
            }

        data_quality = {
            "metric_files": len(frames["metric"]),
            "trace_files": len(frames["trace"]),
            "log_files": len(frames["log"]),
            "metric_series": len(metric_series),
            "series_with_window_rows": sum(1 for item in metric_series if not item.window.empty),
            "trace_rows": int(sum(len(df) for df in trace_windows)),
            "log_rows": int(sum(len(df) for df in log_windows)),
        }

        # Compute per-component span latency statistics from trace data
        # True root causes typically have higher span durations
        # Innovation: Also store individual durations for trace-based reason evidence
        component_latency: Dict[str, Dict[str, float]] = {}
        component_durations: Dict[str, list] = {}
        for df in trace_windows:
            if df.empty or "cmdb_id" not in df.columns or "duration" not in df.columns:
                continue
            for comp, group in df.groupby("cmdb_id"):
                durations = group["duration"].astype(float).tolist()
                if comp not in component_latency:
                    component_latency[comp] = {"sum": 0.0, "count": 0, "max": 0.0}
                    component_durations[comp] = []
                component_latency[comp]["sum"] += sum(durations)
                component_latency[comp]["count"] += len(durations)
                component_latency[comp]["max"] = max(component_latency[comp]["max"], max(durations))
                component_durations[comp].extend(durations)
        # Compute averages and store durations for trace-based reason evidence
        component_latency_stats = {}
        for comp, stats in component_latency.items():
            if stats["count"] > 0:
                component_latency_stats[comp] = {
                    "avg_duration": stats["sum"] / stats["count"],
                    "max_duration": stats["max"],
                    "span_count": stats["count"],
                    "durations": sorted(component_durations.get(comp, [])),
                }

        workspace["data_layer"].update(
            {
                "component_kpi_series": component_kpi_series,
                "full_day_series": {k: v.full for k, v in component_kpi_series.items()},
                "global_thresholds": thresholds,
                "raw_metrics": frames["metric"],
                "raw_traces": trace_windows,
                "raw_logs": log_windows,
                "component_latency_stats": component_latency_stats,
                "data_quality": data_quality,
                "params_used": {
                    "threshold_percentile": threshold_percentile,
                    "low_percentile": low_percentile,
                    "aggregate": aggregate,
                },
            }
        )

        # 醒目打印数据加载结果
        dq = data_quality
        print(f"    [DataAgent] metric文件={dq['metric_files']} trace={dq['trace_files']} log={dq['log_files']}")
        print(f"    [DataAgent] KPI序列数={dq['metric_series']} 窗口内={dq['series_with_window_rows']}")
        print(f"    [DataAgent] trace行数={dq['trace_rows']} log行数={dq['log_rows']}")
        print(f"    [DataAgent] 阈值百分位: P{threshold_percentile}/P{low_percentile}")

        return {"data_quality": data_quality, "thresholds": thresholds}

    def _self_evaluate(
        self,
        result: Dict[str, Any],
        workspace: Dict[str, Any],
        params: Dict[str, Any],
    ) -> Tuple[float, List[str]]:
        quality = result["data_quality"]
        warnings: List[str] = []
        score = 0.0
        if quality["metric_files"] > 0:
            score += 0.35
        else:
            warnings.append("No metric files were loaded.")
        if quality["metric_series"] > 0:
            score += 0.35
        else:
            warnings.append("No component-KPI series could be aggregated.")
        if quality["series_with_window_rows"] > 0:
            score += 0.15
        else:
            warnings.append("No metric series has rows inside the failure window.")
        if quality["trace_files"] > 0:
            score += 0.10
        else:
            warnings.append("Trace data is missing; graph construction may rely on priors.")
        if quality["log_files"] > 0 or workspace["task"]["query"].dataset == "Telecom":
            score += 0.05
        else:
            warnings.append("Log data is missing; reason identification will rely on KPI evidence.")
        return min(1.0, score), warnings
