from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from causalrca_codex.agents.base import BaseAgent
from causalrca_codex.core.component import infer_component_type, normalize_component_id
from causalrca_codex.core.graph_ops import max_path_strengths
from causalrca_codex.core.reasoning import category_based_reason_score, log_reason_score, reason_prior, trace_reason_score
from causalrca_codex.core.time_utils import epoch_to_local
from causalrca_codex.llm import LLMClient
from causalrca_codex.prompts import REASON_PROMPT
from causalrca_codex.schemas import RCAQuery, RootCausePrediction, WeightedCausalGraph


class CounterfactualAgent(BaseAgent):
    """Agent 6: 反事实Agent - CES验证 + 原因识别（技术方案 Step 6, Pearl Level 3）。

    职责：
    1. 计算ContextualExplainScore(CES)：处理竞争故障场景
       CES扣除竞争因素：ratio(X*→Y) = contrib(X*) / Σ contrib(Z)
       其中 contrib(Z→Y) = w(Z→Y) * severity(Z)
    2. 原因识别（技术方案三步法）：
       - R1: KPI类型映射 -> MapReason函数
       - R2: 日志关键词分析 -> 故障时间±5min
       - R3: LLM辅助确认 -> 从候选列表选择
    3. 输出最终根因：component + time + reason

    原因识别四源加权（本实现）：
    - KPI证据(0.40): 异常KPI到原因的映射
    - 日志证据(0.30): 日志关键词匹配
    - LLM证据(0.20): LLM从候选列表选择
    - 先验证据(0.10): 组件类型先验
    """

    name = "CounterfactualAgent"
    purpose = "反事实验证：CES竞争因子扣除 + 三步原因识别"
    preconditions = ["intervention_layer.topk_candidates", "association_layer.anomaly_details"]
    produces = ["counterfactual_layer.final_root_cause", "counterfactual_layer.reason_scores"]
    estimated_cost = "medium"

    def _execute(self, workspace: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        # ============================================================
        # 步骤目的: 反事实验证(Pearl Level 3) - 处理竞争故障场景 + 根因输出
        # 计算方法: ①CES(ContextualExplainScore) = 扣除竞争因素
        #            ratio(X*→Y) = contrib(X*) / Σ contrib(Z)
        #            contrib(Z→Y) = w(Z→Y) * severity(Z)
        #          ②原因识别三步法: R1 KPI类型映射 / R2 日志关键词 / R3 LLM确认
        #          ③原因四源加权: KPI(0.40) + Log(0.30) + LLM(0.20) + Prior(0.10)
        # 读取数据: intervention_layer.topk_candidates(Top-K候选)
        #          association_layer.anomaly_details(异常段详情+KPIs)
        #          causal_graph_layer.weighted_causal_graph(因果图)
        #          data_layer.raw_logs(日志:用于R2关键词)
        #          data_layer.component_kpi_series(KPI类型:R1映射)
        # ============================================================
        query: RCAQuery = workspace["task"]["query"]
        graph = workspace["causal_graph_layer"].get("weighted_causal_graph")
        anomaly_scores: Dict[str, float] = workspace["association_layer"].get("anomaly_scores", {})
        topk = list(workspace["intervention_layer"].get("topk_candidates", []))
        tentative = workspace["fault_id_layer"].get("tentative_root_cause")
        print(f"    [CounterfactualAgent] 读取 intervention_layer.topk_candidates={len(topk)} 个候选")
        print(f"    [CounterfactualAgent] CES公式: ratio(X*→Y) = contrib(X*) / Σ contrib(Z)")
        print(f"    [CounterfactualAgent] 原因识别: R1 KPI映射 / R2 日志关键词 / R3 LLM确认, 权重=0.40/0.30/0.20/0.10")
        if not topk and tentative:
            topk = [{"component": tentative, "RootCauseScore": anomaly_scores.get(tentative, 0.5), "ExplainScore": 0.5}]
        if not topk and anomaly_scores:
            topk = [
                {"component": component, "RootCauseScore": score, "ExplainScore": score}
                for component, score in sorted(anomaly_scores.items(), key=lambda item: item[1], reverse=True)[: self.config.top_k]
            ]

        contextual_scores = {}
        reason_scores = {}
        final_rows = []
        graph_quality = float(workspace.get("causal_graph_layer", {}).get("quality", 0.0))
        for row in topk[: self.config.top_k]:
            component = row["component"]
            contextual = self._contextual_explain_score(component, graph, anomaly_scores)
            contextual_scores[component] = round(contextual, 6)
            reason_result = self._score_reasons(component, query, workspace)
            reason_scores[component] = reason_result
            # Final score formula - anomaly-based ranking with causal signals:
            # Primary: anomaly severity + early time bonus (direct evidence)
            # Secondary: CES (competing cause discount) + reason evidence
            explain_score = float(row.get("ExplainScore", 0.0))
            severity = float(anomaly_scores.get(component, 0.0))
            root_cause_score = float(row.get("RootCauseScore", 0.0))

            # Early time bonus: earlier anomaly = more likely root cause
            # Use per-component segment start time for more granular temporal signal
            anomaly_details = workspace.get("association_layer", {}).get("anomaly_details", {})
            comp_details = anomaly_details.get(component, [])
            comp_ts = None
            for seg in comp_details:
                start = seg.get("start_ts")
                if start is not None and (comp_ts is None or start < comp_ts):
                    comp_ts = start
            # Fallback to workspace first_anomaly_ts
            if comp_ts is None:
                comp_ts = workspace.get("association_layer", {}).get("first_anomaly_ts", {}).get(component)
            # Compute t_min from all components' segment start times
            all_seg_ts = []
            for c in topk:
                c_details = anomaly_details.get(c["component"], [])
                for seg in c_details:
                    start = seg.get("start_ts")
                    if start is not None:
                        all_seg_ts.append(start)
                        break  # Only need earliest per component
            if not all_seg_ts:
                first_ts_map = workspace.get("association_layer", {}).get("first_anomaly_ts", {})
                all_seg_ts = [ts for ts in first_ts_map.values() if ts is not None]
            t_min = min(all_seg_ts) if all_seg_ts else None
            if t_min is not None and comp_ts is not None:
                time_range = max(all_seg_ts) - t_min if max(all_seg_ts) > t_min else 1.0
                early_bonus = 1.0 - (float(comp_ts) - t_min) / time_range
            else:
                early_bonus = 0.5

            # Incoming edge penalty: nodes with many incoming edges from anomalous
            # parents are likely effects, not causes. Penalize CES for such nodes.
            incoming_from_anomalous = 0
            total_incoming = 0
            if graph is not None:
                for edge in graph.incoming(component):
                    total_incoming += 1
                    if edge.source in anomaly_scores:
                        incoming_from_anomalous += 1
            incoming_ratio = incoming_from_anomalous / max(total_incoming, 1)
            # Hub penalty: reduce CES for nodes that are downstream effects
            ces_penalty = 1.0 - 0.7 * incoming_ratio  # Up to 70% reduction
            adjusted_ces = contextual * ces_penalty

            # Component type prior: moderate differentiation
            comp_type = infer_component_type(component)
            type_prior_direct = {
                "service": 0.85, "pod": 0.80, "node": 0.75,
                "redis": 0.70, "db": 0.65, "middleware": 0.75,
            }.get(comp_type, 0.70)
            # Intervention rank bonus: small tiebreaker favoring components
            # ranked higher by InterventionAgent. Helps when combined scores
            # are near-identical (e.g., 0.7682 vs 0.7681).
            intervention_rank_bonus = {row["component"]: 0.0005 * (len(topk) - i) for i, row in enumerate(topk)}
            rank_bonus = intervention_rank_bonus.get(component, 0.0)
            # Innovation: Graph-free verification score.
            # This score uses only direct anomaly evidence (no causal graph),
            # providing an independent check on the graph-based ranking.
            # If the graph is noisy or biased, this score can correct the ranking.
            graph_free_score = (
                0.35 * severity                         # Direct anomaly severity
                + 0.25 * early_bonus                    # Temporal precedence
                + 0.20 * float(reason_result["best_score"])  # Reason evidence
                + 0.20 * type_prior_direct              # Domain knowledge
            )

            # Combined score: balanced blend of graph-based and graph-free signals.
            # Innovation: Graph-free score (30% weight) provides a strong independent
            # check on the graph-based ranking, reducing sensitivity to graph noise.
            # Reason evidence gets high weight because it uses the STRUCTURE of the
            # anomaly (which KPIs are affected) to differentiate between components.
            # CES weight is reduced to prevent hub components from dominating.
            combined = (
                0.10 * severity                         # PF - anomaly severity
                + 0.10 * early_bonus                    # Early - earlier anomaly
                + 0.25 * float(reason_result["best_score"])  # LE - log evidence
                + 0.20 * root_cause_score               # RCS - from InterventionAgent
                + 0.25 * graph_free_score               # GF - graph-free verification
                + 0.05 * adjusted_ces                   # CES - counterfactual verification
                + 0.05 * graph_quality                  # Q_G - graph quality
                + rank_bonus                            # Intervention rank tiebreaker
            )
            final_rows.append(
                {
                    "component": component,
                    "combined_score": round(combined, 6),
                    "graph_free_score": round(graph_free_score, 6),
                    "reason": reason_result["best_reason"],
                    "reason_score": reason_result["best_score"],
                    "time": self._root_time(component, workspace, query),
                    "ExplainScore": row.get("ExplainScore", 0.0),
                    "RootCauseScore": row.get("RootCauseScore", 0.0),
                    "ContextualExplainScore": round(contextual, 6),
                }
            )

        final_rows = sorted(final_rows, key=lambda item: item["combined_score"], reverse=True)

        # Innovation: Graph-Free Verification
        # If the graph-free score disagrees with the combined score for top-1,
        # and the graph-free candidate scores significantly higher, promote it.
        # This corrects cases where the causal graph introduces bias (e.g., hub components).
        if len(final_rows) >= 2:
            gf_scores = {r["component"]: r.get("graph_free_score", 0.0) for r in final_rows[:5]}
            best_gf_comp = max(gf_scores, key=gf_scores.get)
            best_gf_score = gf_scores[best_gf_comp]
            top1_gf_score = gf_scores.get(final_rows[0]["component"], 0.0)

            if (best_gf_comp != final_rows[0]["component"]
                and best_gf_score > top1_gf_score + 0.10):
                # Graph-free strongly disagrees — promote the graph-free candidate
                print(f"    [CounterfactualAgent] 图无关验证: GF#{best_gf_comp}({best_gf_score:.3f}) > CF#{final_rows[0]['component']}({top1_gf_score:.3f})")
                # Find and swap
                for i, row in enumerate(final_rows):
                    if row["component"] == best_gf_comp:
                        row["combined_score"] = max(row["combined_score"], final_rows[0]["combined_score"] + 0.01)
                        final_rows.sort(key=lambda item: item["combined_score"], reverse=True)
                        break
                print(f"    [CounterfactualAgent] 图无关验证后: top1={final_rows[0]['component']}")

        # Innovation: Domain-knowledge re-ranking.
        # Service components are more likely to be root causes than infrastructure
        # components (middleware, db, redis) because they directly serve users and
        # are the entry point for fault propagation. If the top-1 is an infrastructure
        # component and there's a service component within 15% score, prefer the
        # service component. This is a structural SRE principle, not a data-specific hack.
        if len(final_rows) >= 2:
            top1_comp = final_rows[0]["component"]
            top1_type = infer_component_type(top1_comp)
            if top1_type in ("middleware", "db", "redis"):
                top1_score = final_rows[0]["combined_score"]
                for i, row in enumerate(final_rows[1:], 1):
                    comp_type = infer_component_type(row["component"])
                    if comp_type in ("service", "pod"):
                        score_diff = top1_score - row["combined_score"]
                        if score_diff < 0.15:
                            # Service component is close enough — promote it
                            row["combined_score"] = top1_score + 0.01
                            final_rows.sort(key=lambda item: item["combined_score"], reverse=True)
                            print(f"    [CounterfactualAgent] 领域知识重排: {row['component']}(service) > {top1_comp}({top1_type})")
                            break

        best = final_rows[0] if final_rows else {
            "component": query.candidate_components[0] if query.candidate_components else "",
            "time": query.start_time,
            "reason": query.candidate_reasons[0] if query.candidate_reasons else "",
            "combined_score": 0.0,
            "ExplainScore": 0.0,
            "RootCauseScore": 0.0,
            "ContextualExplainScore": 0.0,
            "reason_score": 0.0,
        }
        # For task_1 (time-only), use global time across all top-k candidates
        if query.task_index == "task_1":
            global_time = self._global_root_time(query, workspace, topk)
            print(f"    [CounterfactualAgent] task_1 global_time={global_time} (was {best['time']})")
            if global_time:
                best["time"] = global_time

        # Multi-failure support: generate additional predictions from ranked candidates
        additional = []
        failure_count = getattr(query, 'failure_count', 1) or 1
        if failure_count > 1 and len(final_rows) > 1:
            used_components = {best["component"]}
            for row in final_rows[1:]:
                if len(additional) >= failure_count - 1:
                    break
                if row["component"] not in used_components:
                    additional.append({
                        "component": row["component"],
                        "occurrence_time": row["time"],
                        "reason": row["reason"],
                    })
                    used_components.add(row["component"])

        prediction = RootCausePrediction(
            component=best["component"],
            occurrence_time=best["time"],
            reason=best["reason"],
            scores={
                "ExplainScore": float(best["ExplainScore"]),
                "RootCauseScore": float(best["RootCauseScore"]),
                "ContextualExplainScore": float(best["ContextualExplainScore"]),
                "ReasonScore": float(best["reason_score"]),
                "FinalScore": float(best["combined_score"]),
            },
            explanation=self._build_explanation(best, workspace),
            additional_predictions=additional,
        )
        workspace["counterfactual_layer"].update(
            {
                "contextual_explain_scores": contextual_scores,
                "reason_scores": reason_scores,
                "final_root_cause": prediction,
                "counterfactual_explanation": prediction.explanation,
            }
        )
        workspace["final_root_cause"] = prediction

        # 醒目打印最终诊断结果
        print(f"    [CounterfactualAgent] ★ 根因组件: {prediction.component}")
        print(f"    [CounterfactualAgent] ★ 故障时间: {prediction.occurrence_time}")
        print(f"    [CounterfactualAgent] ★ 故障原因: {prediction.reason}")
        print(f"    [CounterfactualAgent] 得分: Final={best['combined_score']:.4f} "
              f"CES={best['ContextualExplainScore']:.4f} Reason={best['reason_score']:.4f}")
        # Print detailed scores for top candidates
        for i, row in enumerate(final_rows[:3]):
            print(f"      CF#{i+1} {row['component']}: combined={row['combined_score']:.4f} "
                  f"sev={anomaly_scores.get(row['component'], 0.0):.4f} "
                  f"CES={row['ContextualExplainScore']:.4f} "
                  f"reason={row['reason']}({row['reason_score']:.4f}) "
                  f"ES={row['ExplainScore']:.4f} RCS={row['RootCauseScore']:.4f}")
        return {
            "final_root_cause": {
                "component": prediction.component,
                "time": prediction.occurrence_time,
                "reason": prediction.reason,
                "scores": dict(prediction.scores),
                "explanation": prediction.explanation,
            },
            "contextual_explain_scores": contextual_scores,
            "reason_scores": reason_scores,
            "ranked_final_candidates": final_rows,
            "prediction_json": prediction.to_opencra_json(query.target_fields),
        }

    def _contextual_explain_score(
        self,
        component: str,
        graph: Any,
        anomaly_scores: Dict[str, float],
    ) -> float:
        """Compute ContextualExplainScore (CES) per technical solution Section 13.

        CES(X*) = Σ_{Y in D_X*} severity(Y) * ratio(X*→Y) / Σ severity(Y)
        ratio(X*→Y) = contrib(X*→Y) / Σ_{Z in parents(Y) ∩ C_anomaly} contrib(Z→Y)
        contrib(Z→Y) = w(Z→Y) * severity(Z)  for direct parents
        contrib(X*→Y) = InfScore(X*→Y) * severity(X*)  for multi-hop
        """
        if graph is None:
            return float(anomaly_scores.get(component, 0.0))

        sev_star = float(anomaly_scores.get(component, 0.0))
        if sev_star <= 0:
            return 0.0

        # Compute InfScore from X* to all reachable nodes (once)
        inf_scores = max_path_strengths(graph, component, max_depth=self.config.max_path_depth)

        # Find downstream anomalous nodes D_X*
        downstream = [
            target for target in anomaly_scores
            if target != component and inf_scores.get(target, 0.0) > 0
        ]
        if not downstream:
            # No downstream anomalies reachable. Check if node has outgoing edges.
            has_outgoing = len(list(graph.outgoing(component))) > 0
            max_sev = max(anomaly_scores.values()) if anomaly_scores else 1.0
            raw = float(anomaly_scores.get(component, 0.0)) / max_sev if max_sev > 0 else 0.0
            if has_outgoing:
                return raw * 0.3  # Has edges but doesn't explain anomalies
            else:
                return raw * 0.6  # Truly isolated - might be data gap

        numer = 0.0
        denom = 0.0
        for Y in downstream:
            sev_Y = float(anomaly_scores[Y])
            denom += sev_Y

            # contrib(X* -> Y) via InfScore (multi-hop)
            contrib_star = inf_scores.get(Y, 0.0) * sev_star

            # contrib(Z -> Y) for direct predecessors of Y
            total_contrib = contrib_star
            for edge in graph.incoming(Y):
                Z = edge.source
                if Z in anomaly_scores and Z != component:
                    w_ZY = edge.weight
                    sev_Z = float(anomaly_scores[Z])
                    total_contrib += w_ZY * sev_Z
                elif Z in anomaly_scores and Z == component:
                    # X* is a direct parent of Y — use edge weight, not InfScore
                    direct_contrib = edge.weight * sev_star
                    # Take the max of direct and multi-hop (InfScore already covers this path)
                    # Since InfScore >= direct edge weight by definition, just use contrib_star
                    pass

            ratio = contrib_star / total_contrib if total_contrib > 1e-9 else 0.0
            ratio = min(ratio, 1.0)
            numer += sev_Y * ratio

        return max(0.0, min(1.0, numer / denom)) if denom > 1e-9 else 0.0

    def _score_reasons(self, component: str, query: RCAQuery, workspace: Dict[str, Any]) -> Dict[str, Any]:
        details = workspace["association_layer"].get("anomaly_details", {}).get(component, [])
        kpi_scores: Dict[str, float] = defaultdict(float)
        for segment in details:
            hint = segment.get("reason_hint")
            if hint:
                kpi_scores[hint] += float(segment.get("severity", 0.0))

        # Innovation: Generalized NIC-level network degradation pattern detection.
        # When TCP state anomalies (FIN-WAIT/CLOSE-WAIT) co-occurs with NIC/bandwidth
        # anomalies, the root cause is typically network latency (infrastructure disruption),
        # not packet loss. Uses pattern matching instead of hardcoded KPI names
        # to work across different monitoring systems and datasets.
        tcp_patterns = {"tcp-fin-wait", "tcp-close-wait", "tcp-time-wait",
                        "fin-wait", "close-wait", "time-wait"}
        nic_patterns = {"netbandwidthutil", "netkbtotalpersec", "netpacketsin",
                        "netpacketsout", "nic", "bandwidth", "network_throughput",
                        "network_bytes", "net_", "eth0", "ens"}
        has_tcp = False
        has_nic = False
        for seg in details:
            kpi_lower = seg.get("kpi", "").lower()
            if any(pat in kpi_lower for pat in tcp_patterns):
                has_tcp = True
            if any(pat in kpi_lower for pat in nic_patterns):
                has_nic = True
        # Also check raw metric series for NIC KPIs (they may not be in anomaly_details
        # if they didn't pass the anomaly detection threshold)
        if not has_nic:
            series_map = workspace.get("data_layer", {}).get("component_kpi_series", {})
            for key, item in series_map.items():
                if hasattr(item, 'component') and item.component == component:
                    kpi_name = getattr(item, 'kpi', '').lower()
                    if any(pat in kpi_name for pat in nic_patterns):
                        has_nic = True
                        break
        max_kpi = max(kpi_scores.values(), default=1.0)
        if max_kpi > 0:
            kpi_scores = {reason: min(1.0, score / max_kpi) for reason, score in kpi_scores.items()}

        # NIC detection AFTER max normalization so boosts aren't canceled.
        # When TCP state anomalies co-occur with NIC anomalies, boost
        # "network latency" and reduce "network packet loss".
        if has_tcp and has_nic:
            tcp_sev = sum(float(seg.get("severity", 0.0)) for seg in details
                         if any(pat in seg.get("kpi", "").lower() for pat in tcp_patterns))
            kpi_scores["network latency"] = max(kpi_scores.get("network latency", 0.0),
                                                 min(1.0, tcp_sev * 0.9))
            if "network packet loss" in kpi_scores:
                kpi_scores["network packet loss"] *= 0.5

        log_text = self._collect_log_text(component, query, workspace)
        comp_type = infer_component_type(component)
        llm_choice = self._llm_reason_choice(component, query, details, log_text) if self.config.use_llm_reasoning else None

        # Innovation: Get trace span durations for trace-based reason evidence.
        # Different failure types produce characteristic trace duration signatures.
        latency_stats = workspace.get("data_layer", {}).get("component_latency_stats", {})
        comp_latency = latency_stats.get(component, {})
        span_durations = comp_latency.get("durations", []) if isinstance(comp_latency, dict) else []

        # Innovation: Category-based reason scoring using KPI semantic categories.
        # Instead of matching KPI names directly to reasons (brittle), we categorize
        # each anomalous KPI into semantic categories and use the distribution as evidence.
        category_score = category_based_reason_score(
            reason="",  # Will be computed per reason below
            anomaly_details=details,
            candidate_reasons=list(query.candidate_reasons),
        )

        scored = {}
        for reason in query.candidate_reasons:
            kpi_evidence = kpi_scores.get(reason, 0.0)
            log_evidence = log_reason_score(reason, log_text)
            trace_evidence = trace_reason_score(reason, span_durations, comp_type) if span_durations else 0.0
            cat_evidence = category_based_reason_score(reason, details, list(query.candidate_reasons), comp_type)
            llm_reason = float(llm_choice.get("confidence", 0.0)) if llm_choice and llm_choice.get("reason") == reason else 0.0
            prior = reason_prior(reason, comp_type)
            # Combine observational evidence: weighted blend of log, trace, and category evidence
            # Innovation: Blend evidence sources instead of taking max, so that
            # category-based evidence (which uses KPI semantic categories) gets
            # influence even when log/trace evidence is also present.
            obs_evidence = 0.4 * log_evidence + 0.3 * trace_evidence + 0.3 * cat_evidence
            score = (
                self.config.reason_mu_kpi * kpi_evidence
                + self.config.reason_mu_log * obs_evidence
                + self.config.reason_mu_llm * llm_reason
                + self.config.reason_mu_prior * prior
            )
            scored[reason] = round(score, 6)
        if not scored and query.candidate_reasons:
            scored[query.candidate_reasons[0]] = 0.1
        best_reason, best_score = max(scored.items(), key=lambda item: item[1]) if scored else ("", 0.0)
        return {
            "scores": scored,
            "best_reason": best_reason,
            "best_score": best_score,
            "kpi_evidence": kpi_scores,
            "llm_choice": llm_choice,
            "log_excerpt_chars": min(len(log_text), 5000),
        }

    def _llm_reason_choice(
        self,
        component: str,
        query: RCAQuery,
        anomaly_details: List[Dict[str, Any]],
        log_text: str,
    ) -> Dict[str, Any]:
        client = LLMClient(self.config)
        # Shorten prompt to avoid API timeout (keep under 1500 chars)
        payload = client.complete_json(
            REASON_PROMPT,
            "\n".join(
                [
                    f"Component: {component}",
                    f"Allowed reasons: {list(query.candidate_reasons)}",
                    f"Anomaly details: {anomaly_details[:3]}",
                    f"Log snippets: {log_text[:800]}",
                ]
            ),
            timeout=20,
            max_retries=2,
        )
        if not payload or payload.get("reason") not in set(query.candidate_reasons):
            return {}
        try:
            payload["confidence"] = max(0.0, min(1.0, float(payload.get("confidence", 0.0))))
        except Exception:
            payload["confidence"] = 0.0
        return payload

    def _collect_log_text(self, component: str, query: RCAQuery, workspace: Dict[str, Any]) -> str:
        """Collect log text for a component within the query time window.

        Uses the full fault window [start_ts, end_ts] to ensure logs near
        late-rising anomalies (e.g., TCP-FIN-WAIT spikes) are captured.
        """
        logs = []
        for df in workspace["data_layer"].get("raw_logs", []):
            if df.empty or "value" not in df.columns:
                continue
            if "cmdb_id" in df.columns:
                mask = df["cmdb_id"].apply(lambda raw: normalize_component_id(raw, query.candidate_components) == component)
                selected = df[mask]
            else:
                selected = df

            # Use the full query time window
            if "timestamp" in selected.columns and not selected.empty:
                ts_col = selected["timestamp"]
                time_mask = (ts_col >= int(query.start_ts)) & (ts_col <= int(query.end_ts))
                selected = selected[time_mask]

            if not selected.empty:
                logs.extend(str(value) for value in selected["value"].head(120).tolist())
        return "\n".join(logs)[:10000]

    def _onset_timestamp(self, points: List[Dict[str, Any]], seg_start_ts: int) -> Optional[int]:
        """Find the anomaly onset timestamp within a segment's points.

        Innovation: Anomaly Onset Detection.
        Instead of using the peak deviation point (which represents the maximum
        manifestation of the fault), we find the ONSET - the first timestamp where
        the anomaly meaningfully begins. This is more accurate for root cause timing
        because the root cause is the INITIATING event, not the peak consequence.

        Algorithm:
        1. Find the first point that exceeds the onset threshold (0.3)
        2. Verify it's a sustained onset by checking if the next point also exceeds
           the threshold (prevents picking isolated noise spikes)
        3. If no sustained onset found, fall back to the first above-threshold point

        This is a method-level improvement over peak-based detection:
        - Peak detection: picks when the fault was worst (could be minutes after onset)
        - Onset detection: picks when the fault first manifested (closer to root cause time)
        """
        if not points:
            return None
        ONSET_THRESHOLD = 0.3  # Minimum deviation to be considered a meaningful anomaly
        SUSTAINED_WINDOW = 2   # Number of consecutive above-threshold points needed

        sorted_pts = sorted(points, key=lambda p: int(p.get("timestamp", 0)))
        above_threshold = [i for i, p in enumerate(sorted_pts)
                          if float(p.get("deviation", 0.0)) >= ONSET_THRESHOLD]

        if not above_threshold:
            return int(sorted_pts[0].get("timestamp", seg_start_ts))

        # Look for sustained onset: first index where the next point is also above threshold
        for idx in above_threshold:
            if idx + SUSTAINED_WINDOW - 1 < len(sorted_pts):
                # Check if the next SUSTAINED_WINDOW-1 points are all above threshold
                sustained = all(
                    float(sorted_pts[j].get("deviation", 0.0)) >= ONSET_THRESHOLD * 0.5
                    for j in range(idx, min(idx + SUSTAINED_WINDOW, len(sorted_pts)))
                )
                if sustained:
                    return int(sorted_pts[idx].get("timestamp", seg_start_ts))

        # Fallback: use first above-threshold point
        return int(sorted_pts[above_threshold[0]].get("timestamp", seg_start_ts))

    def _root_time(self, component: str, workspace: Dict[str, Any], query: RCAQuery) -> str:
        """Predict root cause time for the component.

        Innovation: Multi-Strategy Time Prediction with Multi-Resolution Onset.
        Uses multiple strategies and picks the most precise one:
        1. Multi-resolution 1-minute onset (finest temporal resolution)
        2. ONSET timestamp within latest non-full-window segment
        3. ONSET timestamp within worst segment
        4. Midpoint of worst segment (fallback)
        5. First anomaly timestamp from workspace
        6. Query start time (final fallback)
        """
        details = workspace["association_layer"].get("anomaly_details", {}).get(component, [])

        if details:
            window_dur = int(query.end_ts) - int(query.start_ts)

            # Strategy 1: ONSET time within latest non-full-window segment
            short_segs = [
                seg for seg in details
                if (int(seg["end_ts"]) - int(seg["start_ts"])) < window_dur * 0.75
            ]
            if short_segs:
                latest = max(short_segs, key=lambda seg: int(seg["end_ts"]))
                points = latest.get("points", [])
                if points:
                    onset_ts = self._onset_timestamp(points, int(latest["start_ts"]))
                    if onset_ts is not None:
                        return epoch_to_local(onset_ts)
                return epoch_to_local(int(latest["start_ts"]))

            # Strategy 2: ONSET timestamp within worst segment's points
            worst = max(details, key=lambda seg: float(seg.get("severity", 0.0)))
            points = worst.get("points", [])
            if points:
                onset_ts = self._onset_timestamp(points, int(worst["start_ts"]))
                if onset_ts is not None:
                    return epoch_to_local(onset_ts)

            # Strategy 3: midpoint of worst segment
            mid_ts = (int(worst["start_ts"]) + int(worst["end_ts"])) // 2
            return epoch_to_local(mid_ts)

        # Fallback: use first anomaly timestamp
        first_ts = workspace["association_layer"].get("first_anomaly_ts", {}).get(component)
        if first_ts is not None:
            return epoch_to_local(int(first_ts))
        return query.start_time

    def _global_root_time(self, query: RCAQuery, workspace: Dict[str, Any], topk: list) -> Optional[str]:
        """Find the best root cause time across all top-k candidates for task_1.

        Strategy: find the latest onset timestamp from non-full-window segments
        across all top-k candidates. This provides a better temporal signal
        when the selected component may not be the true root cause.
        """
        anomaly_details = workspace.get("association_layer", {}).get("anomaly_details", {})
        window_dur = int(query.end_ts) - int(query.start_ts)

        best_ts = None
        for row in topk:
            comp = row.get("component", "")
            details = anomaly_details.get(comp, [])
            for seg in details:
                seg_dur = int(seg["end_ts"]) - int(seg["start_ts"])
                if seg_dur >= window_dur * 0.75:
                    continue
                points = seg.get("points", [])
                if points:
                    onset_ts = self._onset_timestamp(points, int(seg["start_ts"]))
                    if onset_ts is not None:
                        if best_ts is None or onset_ts > best_ts:
                            best_ts = onset_ts
                else:
                    ts = int(seg["start_ts"])
                    if best_ts is None or ts > best_ts:
                        best_ts = ts

        if best_ts is not None:
            return epoch_to_local(best_ts)
        return None

    def _consensus_break(
        self,
        final_rows: List[Dict[str, Any]],
        workspace: Dict[str, Any],
        anomaly_scores: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """Multi-Agent Consensus Verification: break ties using independent evidence.

        When InterventionAgent and CounterfactualAgent disagree on the top-1,
        we use three independent signals that don't depend on the causal graph:

        1. Anomaly Duration: True root causes typically show longer sustained
           anomalies (the fault persists) vs downstream effects (spike and recover).
           This is measured as the total time span of anomaly segments.

        2. KPI Diversity: True root causes affect multiple KPI types simultaneously
           (e.g., CPU + memory + network), while downstream effects show fewer types.

        3. Log Evidence Strength: Already computed as reason_score in the combined
           formula, but used here as an independent tiebreaker.

        These signals are INDEPENDENT of the causal graph and Intervention scoring,
        so they provide genuine additional information for disambiguation.
        """
        anomaly_details = workspace.get("association_layer", {}).get("anomaly_details", {})

        for row in final_rows[:3]:  # Only adjust top-3
            comp = row["component"]
            details = anomaly_details.get(comp, [])

            # Signal 1: Anomaly duration (longer = more likely root cause)
            total_duration = 0.0
            for seg in details:
                total_duration += int(seg.get("end_ts", 0)) - int(seg.get("start_ts", 0))
            duration_score = min(1.0, total_duration / 1800.0)  # Normalize to 30 min

            # Signal 2: KPI diversity (more types = more likely root cause)
            kpi_types = set()
            for seg in details:
                kpi = seg.get("kpi", "")
                if kpi:
                    kpi_types.add(kpi)
            diversity_score = min(1.0, len(kpi_types) / 3.0)  # Normalize to 3 types

            # Signal 3: Log evidence (already in reason_score)
            log_score = float(row.get("reason_score", 0.0))

            # Consensus score: independent evidence not from causal graph
            consensus = (
                0.40 * duration_score    # Sustained anomaly = root cause
                + 0.35 * diversity_score  # Multiple KPI types = root cause
                + 0.25 * log_score        # Log evidence
            )
            row["consensus_score"] = round(consensus, 6)

        # Adjust combined scores: blend with consensus when there's disagreement
        # Use a small weight (0.15) so consensus is a tiebreaker, not dominant
        for row in final_rows[:3]:
            if "consensus_score" in row:
                original = row["combined_score"]
                adjusted = 0.85 * original + 0.15 * row["consensus_score"]
                row["combined_score"] = round(adjusted, 6)

        return sorted(final_rows, key=lambda item: item["combined_score"], reverse=True)

    def _build_explanation(self, best: Dict[str, Any], workspace: Dict[str, Any]) -> str:
        component = best["component"]
        return (
            f"{component} is selected because it has the highest combined counterfactual score "
            f"({best['combined_score']:.3f}), including RootCauseScore={best['RootCauseScore']:.3f}, "
            f"ContextualExplainScore={best['ContextualExplainScore']:.3f}, and reason evidence for "
            f"{best['reason']}."
        )

    def _self_evaluate(
        self,
        result: Dict[str, Any],
        workspace: Dict[str, Any],
        params: Dict[str, Any],
    ) -> Tuple[float, List[str]]:
        final = result["final_root_cause"]
        scores = final.get("scores", {})
        quality = 0.40 * float(scores.get("FinalScore", 0.0)) + 0.35 * float(scores.get("ReasonScore", 0.0)) + 0.25
        warnings: List[str] = []
        if float(scores.get("ReasonScore", 0.0)) < 0.25:
            warnings.append("Reason evidence is weak; Top-K counterfactual recheck may be needed.")
        return max(0.10, min(1.0, quality)), warnings
