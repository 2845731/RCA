from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from causalrca_codex.agents.base import BaseAgent
from causalrca_codex.core.component import infer_component_type
from causalrca_codex.core.evidence import build_component_profiles
from causalrca_codex.core.reasoning import absolute_domain_deviation, is_low_sensitive_kpi, reason_hint
from causalrca_codex.core.time_utils import epoch_to_local
from causalrca_codex.schemas import AnomalySegment, MetricSeries, RCAQuery


class AssociationAgent(BaseAgent):
    """Agent 2: 关联分析Agent - 异常检测（技术方案 Step 2, Pearl Level 1）。

    职责：在故障窗口内检测异常组件。
    核心算法：
    1. 异常点检测：x(t) 超过上阈值(P_q)或低于下阈值(P_{100-q}) 为异常
    2. 故障识别：连续异常点构成"故障段"，过滤条件：
       - F1（噪声）：连续异常点数 >= min_fault_points
       - F2（假阳性）：偏差比 >= beta_min
    3. 聚合：每个组件的严重度 = 1 - exp(-sum(deviations))

    可调参数：
    - threshold_percentile (q): 默认95, 范围[80,99]
    - min_fault_points (d_min): 默认2, 范围[1,5]
    - beta_min: 默认0.5, 范围[0.1,2.0]
    """

    name = "AssociationAgent"
    purpose = "异常检测：阈值偏差 + 连续故障段过滤 + 严重度评分"
    preconditions = ["data_layer.component_kpi_series"]
    produces = ["association_layer.candidate_set", "association_layer.anomaly_details", "association_layer.anomaly_scores"]
    tunable_params = {
        "threshold_percentile": 95.0,
        "min_fault_points": 2,
        "beta_min": 0.5,
        "severity_threshold": 0.05,
    }

    def _execute(self, workspace: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        # ============================================================
        # 步骤目的: 异常检测(Pearl Level 1 关联) - 在故障窗口内识别异常组件
        # 计算方法: ①x(t) 越过上阈值 P_q 或下阈值 P_{100-q} -> 异常点
        #          ②连续 >= min_fault_points 个异常点 -> 故障段(过滤噪声)
        #          ③max(偏差)/KPI中位数 偏差比 >= beta_min(过滤假阳性)
        #          ④严重度 = min(3.0, 相对偏差 × 多KPI因子) / 3.0
        # 读取数据: data_layer.component_kpi_series(每组件KPI时序)
        #          data_layer.global_thresholds(全天阈值 P95/P5)
        #          task.query.candidate_components(候选组件范围)
        # ============================================================
        query: RCAQuery = workspace["task"]["query"]
        min_points = int(params.get("min_fault_points", self.config.min_fault_points))
        beta_min = float(params.get("beta_min", self.config.beta_min))
        severity_threshold = float(params.get("severity_threshold", self.config.severity_threshold))
        candidate_only = bool(params.get("candidate_components_only", True))

        series_map: Dict[str, MetricSeries] = workspace["data_layer"]["component_kpi_series"]
        print(f"    [AssociationAgent] 读取 data_layer.component_kpi_series={len(series_map)} 个时序")
        print(f"    [AssociationAgent] 读取 task.query.candidate_components={len(query.candidate_components)} 个候选")
        print(f"    [AssociationAgent] 阈值: P95/P5, 连续>=min_points={min_points}, 偏差比>=beta_min={beta_min}")
        all_segments: List[AnomalySegment] = []
        intensity_by_component: Dict[str, pd.DataFrame] = {}

        for item in series_map.values():
            if candidate_only and item.component not in set(query.candidate_components):
                continue
            segments, intensity = self._detect_segments(item, query, min_points, beta_min)
            all_segments.extend(segments)
            if intensity is not None and not intensity.empty:
                existing = intensity_by_component.get(item.component)
                if existing is None:
                    intensity_by_component[item.component] = intensity
                else:
                    merged = pd.concat([existing, intensity]).groupby("timestamp", as_index=False)["intensity"].max()
                    intensity_by_component[item.component] = merged

        by_component: Dict[str, List[AnomalySegment]] = defaultdict(list)
        for segment in all_segments:
            by_component[segment.component].append(segment)

        # Collect deviations per KPI across all components for relative comparison
        kpi_deviations: Dict[str, List[float]] = defaultdict(list)
        for segment in all_segments:
            kpi_deviations[segment.kpi].append(max(segment.max_deviation, 0.0))

        # Calculate median deviation per KPI for relative comparison
        kpi_median: Dict[str, float] = {}
        for kpi, devs in kpi_deviations.items():
            sorted_devs = sorted(devs)
            n = len(sorted_devs)
            kpi_median[kpi] = sorted_devs[n // 2] if n > 0 else 1.0

        component_scores_raw = {}
        first_anomaly_ts = {}
        for component, segments in by_component.items():
            # Use relative severity: compare component's deviation vs median for same KPI
            # This preserves the signal when one component has much higher deviation than others
            # e.g., Tomcat04 network deviation=7.5 vs median=1.0 → relative=7.5
            max_relative = 0.0
            distinct_kpis = set()
            for seg in segments:
                dev = max(seg.max_deviation, 0.0)
                median = max(kpi_median.get(seg.kpi, 1.0), 0.01)
                relative = dev / median
                max_relative = max(max_relative, relative)
                distinct_kpis.add(seg.kpi)
            # Multi-KPI bonus: root causes often affect multiple KPIs simultaneously
            # (e.g., CPU + memory + network), while downstream effects show fewer types
            multi_kpi_factor = 1.0 + 0.15 * max(0, len(distinct_kpis) - 1)
            component_scores_raw[component] = max_relative * multi_kpi_factor
            first_anomaly_ts[component] = min(seg.start_ts for seg in segments)

        # Cap relative severities at a reasonable max to prevent outliers
        # but do NOT normalize to 0-1 — preserve the relative magnitude
        # so that components with much higher deviation stand out
        SEVERITY_CAP = 3.0
        component_scores = {
            comp: round(min(SEVERITY_CAP, score) / SEVERITY_CAP, 6)
            for comp, score in component_scores_raw.items()
        }

        candidate_set = [
            component
            for component, score in sorted(component_scores.items(), key=lambda item: item[1], reverse=True)
            if score > severity_threshold
        ][: self.config.max_candidate_components]

        anomaly_details = {
            component: [seg.to_dict() for seg in sorted(segments, key=lambda seg: seg.severity, reverse=True)]
            for component, segments in by_component.items()
            if component in candidate_set
        }
        component_profiles = build_component_profiles(anomaly_details, query.candidate_reasons)

        serializable_intensity = {
            component: frame.to_dict("records")
            for component, frame in intensity_by_component.items()
            if component in candidate_set
        }

        workspace["association_layer"].update(
            {
                "candidate_set": candidate_set,
                "anomaly_details": anomaly_details,
                "anomaly_scores": {component: component_scores[component] for component in candidate_set},
                "first_anomaly_ts": {component: first_anomaly_ts.get(component) for component in candidate_set},
                "component_profiles": component_profiles,
                "component_intensity_series": intensity_by_component,
                "params_used": {
                    "min_fault_points": min_points,
                    "beta_min": beta_min,
                    "severity_threshold": severity_threshold,
                    "candidate_components_only": candidate_only,
                },
            }
        )

        # 醒目打印异常检测结果
        print(f"    [AssociationAgent] 候选组件={len(candidate_set)} | min_points={min_points} beta_min={beta_min}")
        for comp in candidate_set:
            score = component_scores.get(comp, 0)
            seg_count = len(by_component.get(comp, []))
            comp_type = infer_component_type(comp)
            comp_kpis = sorted({seg.kpi for seg in by_component.get(comp, [])})
            print(f"      {comp} ({comp_type}): score={score:.4f} 故障段={seg_count} KPIs={len(comp_kpis)}")
            for kpi in comp_kpis:
                kpi_segs = [s for s in by_component.get(comp, []) if s.kpi == kpi]
                max_dev = max(s.max_deviation for s in kpi_segs)
                print(f"        - {kpi}: 段数={len(kpi_segs)} max_dev={max_dev:.4f}")

        return {
            "candidate_set": candidate_set,
            "anomaly_details": anomaly_details,
            "anomaly_scores": workspace["association_layer"]["anomaly_scores"],
            "first_anomaly_ts": workspace["association_layer"]["first_anomaly_ts"],
            "component_profiles": component_profiles,
            "component_intensity_series": serializable_intensity,
        }

    def _detect_segments(
        self,
        item: MetricSeries,
        query: RCAQuery,
        min_points: int,
        beta_min: float,
    ) -> Tuple[List[AnomalySegment], Optional[pd.DataFrame]]:
        window = item.window.copy()
        if window.empty:
            return [], None

        # Detect persistent saturation: if >80% of window values are above 90% of threshold
        # This catches cases where KPI is consistently near ceiling (e.g., memory at 98% all day)
        saturation_deviation = 0.0
        if len(window) >= 3:
            values = window["value"].astype(float)
            kpi_lower = item.kpi.lower()
            is_mem = any(key in kpi_lower for key in ["mem", "memory", "heap"])
            is_cpu = "cpu" in kpi_lower and any(key in kpi_lower for key in ["util", "percent", "pct"])
            if is_mem and item.median >= 90:
                high_ratio = (values >= 95).sum() / len(values)
                if high_ratio >= 0.7:
                    saturation_deviation = min(1.0, 0.30 + (values.mean() - 90.0) / 15.0)
            elif is_cpu and item.median >= 80:
                high_ratio = (values >= 90).sum() / len(values)
                if high_ratio >= 0.7:
                    saturation_deviation = min(1.0, 0.25 + (values.mean() - 80.0) / 20.0)

        # Check if this is a percent/rate KPI (bounded 0-100) vs absolute KPI
        kpi_lower = item.kpi.lower()
        is_percent_kpi = any(key in kpi_lower for key in ["percent", "pct", "perc", "util", "rate"])

        rows = []
        for _, row in window.sort_values("timestamp").iterrows():
            value = float(row["value"])
            raw_high = max(0.0, (value - item.threshold_high) / max(abs(item.threshold_high), item.scale, 1e-6))
            raw_low = max(0.0, (item.threshold_low - value) / max(abs(item.threshold_low), item.scale, 1e-6))
            if not is_low_sensitive_kpi(item.kpi):
                raw_low *= 0.25
            # Scale down percentile-based deviations to prevent outlier domination
            # Large deviations from small-scale KPIs get compressed
            high_dev = 0.3 * raw_high if raw_high > 1.0 else raw_high
            low_dev = 0.3 * raw_low if raw_low > 1.0 else raw_low
            # Domain deviation (already capped at 1.0)
            domain_dev = absolute_domain_deviation(item.kpi, value, item.median, item.threshold_high)
            high_dev = max(high_dev, domain_dev)
            # Add persistent saturation deviation
            if saturation_deviation > 0:
                high_dev = max(high_dev, saturation_deviation)
            direction = "high" if high_dev >= low_dev else "low"
            deviation = max(high_dev, low_dev)
            if deviation > 0:
                rows.append(
                    {
                        "timestamp": int(row["timestamp"]),
                        "value": value,
                        "deviation": float(deviation),
                        "direction": direction,
                        "threshold": item.threshold_high if direction == "high" else item.threshold_low,
                    }
                )
        intensity = pd.DataFrame(
            [{"timestamp": row["timestamp"], "intensity": row["deviation"]} for row in rows]
        )
        selected = [row for row in rows if row["deviation"] >= beta_min]
        if not selected:
            return [], intensity

        full_times = item.full["timestamp"].sort_values().diff().dropna()
        median_gap = int(full_times.median()) if not full_times.empty else 60
        max_gap = max(180, median_gap * 3)
        groups: List[List[Dict[str, Any]]] = []
        current = [selected[0]]
        for row in selected[1:]:
            if row["timestamp"] - current[-1]["timestamp"] <= max_gap:
                current.append(row)
            else:
                groups.append(current)
                current = [row]
        groups.append(current)

        hint = reason_hint(query.candidate_reasons, item.kpi, item.file_name)
        segments = []
        for group in groups:
            if len(group) < min_points:
                continue
            peak = max(group, key=lambda row: row["deviation"])
            # Severity = max deviation across the fault segment (technical solution:
            # beta(C_i,k,fault) = max(values) - threshold / threshold)
            severity = float(peak["deviation"])
            segments.append(
                AnomalySegment(
                    component=item.component,
                    component_type=item.component_type,
                    kpi=item.kpi,
                    file_name=item.file_name,
                    start_ts=int(group[0]["timestamp"]),
                    end_ts=int(group[-1]["timestamp"]),
                    start_time=epoch_to_local(int(group[0]["timestamp"])),
                    end_time=epoch_to_local(int(group[-1]["timestamp"])),
                    direction=str(peak["direction"]),
                    peak_value=float(peak["value"]),
                    threshold=float(peak["threshold"]),
                    max_deviation=float(peak["deviation"]),
                    severity=round(float(severity), 6),
                    reason_hint=hint,
                    points=group[:10],
                )
            )
        return segments, intensity

    def _self_evaluate(
        self,
        result: Dict[str, Any],
        workspace: Dict[str, Any],
        params: Dict[str, Any],
    ) -> Tuple[float, List[str]]:
        n = len(result["candidate_set"])
        if n == 0:
            return 0.10, ["No anomalous component detected; threshold may be too strict or window misaligned."]
        if n == 1:
            return 0.95, ["Single anomalous component; direct localization may be possible."]
        if 2 <= n <= 10:
            return 0.90, []
        if 11 <= n <= 30:
            return 0.70, ["Many anomalous components; downstream coarse filtering is required."]
        return 0.40, ["Too many anomalous components; threshold may be too loose."]
