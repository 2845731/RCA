from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from causalrca_codex.agents.base import BaseAgent
from causalrca_codex.core.component import infer_component_type, normalize_component_id
from causalrca_codex.core.evidence import reason_family, semantic_match_score
from causalrca_codex.core.graph_ops import max_path_strengths
from causalrca_codex.core.reasoning import log_reason_score, reason_prior
from causalrca_codex.core.time_utils import epoch_to_local
from causalrca_codex.llm import LLMClient
from causalrca_codex.prompts import REASON_PROMPT
from causalrca_codex.schemas import RCAQuery, RootCausePrediction


class CounterfactualAgent(BaseAgent):
    """Agent 6: counterfactual verification and final answer selection.

    The final candidate row contains only four scores:
    - RootCauseScore: intervention-stage causal ranking.
    - CounterfactualScore: CES after competing-parent discount.
    - ReasonScore: evidence for the selected reason.
    - FinalScore: final arbitration score.
    """

    name = "CounterfactualAgent"
    purpose = "Verify candidates with CES, select reason and final root cause"
    preconditions = ["intervention_layer.topk_candidates", "association_layer.anomaly_details"]
    produces = ["counterfactual_layer.final_root_cause", "counterfactual_layer.reason_scores"]
    estimated_cost = "medium"

    def _execute(self, workspace: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        query: RCAQuery = workspace["task"]["query"]
        graph = workspace["causal_graph_layer"].get("weighted_causal_graph")
        anomaly_scores: Dict[str, float] = workspace["association_layer"].get("anomaly_scores", {})
        topk = list(workspace["intervention_layer"].get("topk_candidates", []))
        tentative = workspace["fault_id_layer"].get("tentative_root_cause")

        print(f"    [CounterfactualAgent] topk_candidates={len(topk)}")
        print("    [CounterfactualAgent] FinalScore = 0.55*RCS + 0.25*CFS + 0.20*Reason")

        if not topk and tentative:
            topk = [{"component": tentative, "RootCauseScore": anomaly_scores.get(tentative, 0.5), "ExplainScore": 0.5}]
        if not topk and anomaly_scores:
            topk = [
                {"component": component, "RootCauseScore": score, "ExplainScore": score}
                for component, score in sorted(anomaly_scores.items(), key=lambda item: item[1], reverse=True)[: self.config.top_k]
            ]

        counterfactual_scores: Dict[str, float] = {}
        reason_scores: Dict[str, Dict[str, Any]] = {}
        final_rows: List[Dict[str, Any]] = []

        for row in topk:
            component = str(row.get("component", ""))
            if not component:
                continue
            contextual = self._contextual_explain_score(component, graph, anomaly_scores)
            counterfactual_score = self._discount_downstream_effect(component, contextual, graph, anomaly_scores)
            counterfactual_scores[component] = round(counterfactual_score, 6)

            reason_result = self._score_reasons(component, query, workspace)
            reason_score = float(reason_result["best_score"])
            reason_scores[component] = {
                "best_reason": reason_result["best_reason"],
                "ReasonScore": round(reason_score, 6),
            }

            root_score = float(row.get("RootCauseScore", 0.0))
            final_score = max(0.0, min(1.0, 0.55 * root_score + 0.25 * counterfactual_score + 0.20 * reason_score))

            final_rows.append(
                {
                    "component": component,
                    "FinalScore": round(final_score, 6),
                    "RootCauseScore": round(root_score, 6),
                    "CounterfactualScore": round(counterfactual_score, 6),
                    "ReasonScore": round(reason_score, 6),
                    "ExplainScore": float(row.get("ExplainScore", 0.0)),
                    "reason": reason_result["best_reason"],
                    "time": self._root_time(component, workspace, query, reason_result["best_reason"]),
                }
            )

        final_rows = sorted(final_rows, key=lambda item: item["FinalScore"], reverse=True)
        best = final_rows[0] if final_rows else {
            "component": query.candidate_components[0] if query.candidate_components else "",
            "time": query.start_time,
            "reason": query.candidate_reasons[0] if query.candidate_reasons else "",
            "FinalScore": 0.0,
            "RootCauseScore": 0.0,
            "CounterfactualScore": 0.0,
            "ReasonScore": 0.0,
            "ExplainScore": 0.0,
        }

        causes = self._select_final_causes(final_rows, query, workspace, topk)
        if causes:
            best = {**best, **causes[0]}

        prediction = RootCausePrediction(
            component=best["component"],
            occurrence_time=best["time"],
            reason=best["reason"],
            causes=[
                {
                    "component": cause.get("component", ""),
                    "time": cause.get("time", ""),
                    "reason": cause.get("reason", ""),
                }
                for cause in causes
            ],
            scores={
                "RootCauseScore": float(best["RootCauseScore"]),
                "CounterfactualScore": float(best["CounterfactualScore"]),
                "ReasonScore": float(best["ReasonScore"]),
                "FinalScore": float(best["FinalScore"]),
            },
            explanation=self._build_explanation(best),
        )

        workspace["counterfactual_layer"].update(
            {
                "counterfactual_scores": counterfactual_scores,
                "reason_scores": reason_scores,
                "final_root_cause": prediction,
                "counterfactual_explanation": prediction.explanation,
            }
        )
        workspace["final_root_cause"] = prediction

        print(f"    [CounterfactualAgent] root component: {prediction.component}")
        print(f"    [CounterfactualAgent] root time     : {prediction.occurrence_time}")
        print(f"    [CounterfactualAgent] root reason   : {prediction.reason}")
        print(
            f"    [CounterfactualAgent] score: Final={best['FinalScore']:.4f} "
            f"RCS={best['RootCauseScore']:.4f} CFS={best['CounterfactualScore']:.4f} "
            f"Reason={best['ReasonScore']:.4f}"
        )
        for i, row in enumerate(final_rows[:3]):
            print(
                f"      CF#{i + 1} {row['component']}: Final={row['FinalScore']:.4f} "
                f"RCS={row['RootCauseScore']:.4f} CFS={row['CounterfactualScore']:.4f} "
                f"Reason={row['reason']}({row['ReasonScore']:.4f})"
            )

        return {
            "final_root_cause": {
                "component": prediction.component,
                "time": prediction.occurrence_time,
                "reason": prediction.reason,
                "scores": dict(prediction.scores),
                "explanation": prediction.explanation,
            },
            "counterfactual_scores": counterfactual_scores,
            "reason_scores": reason_scores,
            "ranked_final_candidates": final_rows,
            "prediction_json": prediction.to_opencra_json(query.target_fields),
        }

    def _discount_downstream_effect(
        self,
        component: str,
        contextual: float,
        graph: Any,
        anomaly_scores: Dict[str, float],
    ) -> float:
        if graph is None:
            return max(0.0, min(1.0, contextual))
        total_incoming = 0
        incoming_from_anomalous = 0
        for edge in graph.incoming(component):
            total_incoming += 1
            if edge.source in anomaly_scores:
                incoming_from_anomalous += 1
        incoming_ratio = incoming_from_anomalous / max(total_incoming, 1)
        return max(0.0, min(1.0, contextual * (1.0 - 0.7 * incoming_ratio)))

    def _contextual_explain_score(
        self,
        component: str,
        graph: Any,
        anomaly_scores: Dict[str, float],
    ) -> float:
        if graph is None:
            return float(anomaly_scores.get(component, 0.0))

        sev_star = float(anomaly_scores.get(component, 0.0))
        if sev_star <= 0:
            return 0.0

        inf_scores = max_path_strengths(graph, component, max_depth=self.config.max_path_depth)
        downstream = [
            target for target in anomaly_scores
            if target != component and inf_scores.get(target, 0.0) > 0
        ]
        if not downstream:
            max_sev = max(anomaly_scores.values()) if anomaly_scores else 1.0
            raw = sev_star / max(max_sev, 1e-9)
            return max(0.0, min(1.0, raw * 0.5))

        numer = 0.0
        denom = 0.0
        for target in downstream:
            sev_target = float(anomaly_scores[target])
            denom += sev_target
            contrib_star = inf_scores.get(target, 0.0) * sev_star
            total_contrib = contrib_star
            for edge in graph.incoming(target):
                source = edge.source
                if source in anomaly_scores and source != component:
                    total_contrib += float(edge.weight) * float(anomaly_scores[source])
            ratio = contrib_star / total_contrib if total_contrib > 1e-9 else 0.0
            numer += sev_target * min(ratio, 1.0)
        return max(0.0, min(1.0, numer / denom)) if denom > 1e-9 else 0.0

    def _score_reasons(self, component: str, query: RCAQuery, workspace: Dict[str, Any]) -> Dict[str, Any]:
        details = workspace["association_layer"].get("anomaly_details", {}).get(component, [])
        profile = workspace.get("association_layer", {}).get("component_profiles", {}).get(component, {})
        profile_reason = profile.get("reason_evidence", {})
        log_text = self._collect_log_text(component, query, workspace)
        comp_type = infer_component_type(component)
        llm_choice = self._llm_reason_choice(component, query, details, log_text) if self.config.use_llm_reasoning else None

        kpi_scores: Dict[str, float] = defaultdict(float)
        for segment in details:
            severity = float(segment.get("severity", 0.0))
            hint = segment.get("reason_hint")
            if hint:
                kpi_scores[hint] += severity
            for reason in query.candidate_reasons:
                match = semantic_match_score(reason, segment.get("kpi", ""), segment.get("file_name", ""))
                if match > 0:
                    kpi_scores[reason] += severity * match
        max_kpi = max(kpi_scores.values(), default=1.0)
        if max_kpi > 0:
            kpi_scores = {reason: min(1.0, score / max_kpi) for reason, score in kpi_scores.items()}

        scored = {}
        for reason in query.candidate_reasons:
            kpi_evidence = max(kpi_scores.get(reason, 0.0), float(profile_reason.get(reason, 0.0)))
            log_evidence = log_reason_score(reason, log_text)
            llm_evidence = float(llm_choice.get("confidence", 0.0)) if llm_choice and llm_choice.get("reason") == reason else 0.0
            prior = reason_prior(reason, comp_type)
            reliability = max(kpi_evidence, log_evidence, llm_evidence)
            prior_weight = 0.10 + 0.25 * (1.0 - reliability)
            evidence_weight = 1.0 - prior_weight
            scored[reason] = round(
                evidence_weight * (0.70 * kpi_evidence + 0.25 * log_evidence + 0.05 * llm_evidence)
                + prior_weight * prior,
                6,
            )
        if not scored and query.candidate_reasons:
            scored[query.candidate_reasons[0]] = 0.1
        best_reason, best_score = max(scored.items(), key=lambda item: item[1]) if scored else ("", 0.0)
        return {"scores": scored, "best_reason": best_reason, "best_score": best_score}

    def _llm_reason_choice(
        self,
        component: str,
        query: RCAQuery,
        anomaly_details: List[Dict[str, Any]],
        log_text: str,
    ) -> Dict[str, Any]:
        client = LLMClient(self.config)
        payload = client.complete_json(
            REASON_PROMPT,
            "\n".join(
                [
                    f"Component: {component}",
                    f"Allowed reasons: {list(query.candidate_reasons)}",
                    f"Anomaly details: {anomaly_details[:5]}",
                    f"Log snippets: {log_text[:3000]}",
                ]
            ),
        )
        if not payload or payload.get("reason") not in set(query.candidate_reasons):
            return {}
        try:
            payload["confidence"] = max(0.0, min(1.0, float(payload.get("confidence", 0.0))))
        except Exception:
            payload["confidence"] = 0.0
        return payload

    def _collect_log_text(self, component: str, query: RCAQuery, workspace: Dict[str, Any]) -> str:
        logs: List[str] = []
        for df in workspace["data_layer"].get("raw_logs", []):
            if getattr(df, "empty", True) or "value" not in df.columns:
                continue
            if "cmdb_id" in df.columns:
                mask = df["cmdb_id"].apply(lambda raw: normalize_component_id(raw, query.candidate_components) == component)
                selected = df[mask]
            else:
                selected = df
            if "timestamp" in selected.columns and not selected.empty:
                time_mask = (selected["timestamp"] >= int(query.start_ts)) & (selected["timestamp"] <= int(query.end_ts))
                selected = selected[time_mask]
            if not selected.empty:
                logs.extend(str(value) for value in selected["value"].head(120).tolist())
        return "\n".join(logs)[:10000]

    def _root_time(self, component: str, workspace: Dict[str, Any], query: RCAQuery, reason: str = "") -> str:
        details = workspace["association_layer"].get("anomaly_details", {}).get(component, [])
        if details:
            window_dur = int(query.end_ts) - int(query.start_ts)
            scored_segments = []
            for seg in details:
                seg_dur = int(seg["end_ts"]) - int(seg["start_ts"])
                compactness = 1.0 - min(1.0, seg_dur / max(window_dur, 1))
                match = semantic_match_score(reason, seg.get("kpi", ""), seg.get("file_name", "")) if reason else 0.5
                hint_match = 1.0 if reason and seg.get("reason_hint") == reason else 0.0
                severity = float(seg.get("severity", 0.0))
                score = severity * (0.55 * max(match, hint_match) + 0.30 * compactness + 0.15)
                scored_segments.append((score, seg))
            scored_segments.sort(key=lambda item: item[0], reverse=True)
            for _, seg in scored_segments:
                ts = self._segment_event_ts(seg)
                if ts is not None:
                    return epoch_to_local(ts)

        first_ts = workspace["association_layer"].get("first_anomaly_ts", {}).get(component)
        if first_ts is not None:
            return epoch_to_local(int(first_ts))
        return query.start_time

    def _select_final_causes(
        self,
        final_rows: List[Dict[str, Any]],
        query: RCAQuery,
        workspace: Dict[str, Any],
        topk: List[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        desired = max(1, int(getattr(query, "failure_count", 1) or 1))
        if query.task_index == "task_1":
            times = (
                [self._global_root_time(query, workspace, topk)]
                if desired == 1
                else self._global_root_times(query, workspace, topk, desired)
            )
            times = [time_value for time_value in times if time_value]
            if times:
                seed = final_rows[0] if final_rows else {}
                return [
                    {
                        "component": seed.get("component", ""),
                        "time": time_value,
                        "reason": seed.get("reason", ""),
                    }
                    for time_value in times[:desired]
                ]

        if desired == 1:
            if not final_rows:
                return []
            return [{
                "component": final_rows[0].get("component", ""),
                "time": final_rows[0].get("time", query.start_time),
                "reason": final_rows[0].get("reason", ""),
            }]

        candidates: List[Dict[str, Any]] = [dict(row) for row in final_rows]
        anomaly_details = workspace.get("association_layer", {}).get("anomaly_details", {})
        by_component = {row.get("component"): row for row in final_rows}
        for row in topk:
            component = row.get("component", "")
            base = by_component.get(component)
            if not base:
                continue
            for seg in anomaly_details.get(component, []):
                ts = self._segment_event_ts(seg)
                if ts is None:
                    continue
                candidates.append({
                    **base,
                    "time": epoch_to_local(ts),
                    "event_ts": ts,
                    "segment_family": reason_family(seg.get("reason_hint", "") or base.get("reason", "")),
                    "FinalScore": float(base.get("FinalScore", 0.0)) * (0.75 + 0.25 * float(seg.get("severity", 0.0))),
                })

        candidates = sorted(candidates, key=lambda item: float(item.get("FinalScore", 0.0)), reverse=True)
        min_gap = max(60.0, (int(query.end_ts) - int(query.start_ts)) / max(desired * 2, 1))
        selected: List[Dict[str, Any]] = []
        for candidate in candidates:
            cand_ts = self._parse_time_to_epoch(candidate.get("time", ""), query)
            if cand_ts is None:
                continue
            if any(abs(cand_ts - self._parse_time_to_epoch(item.get("time", ""), query)) < min_gap for item in selected):
                continue
            selected.append(candidate)
            if len(selected) >= desired:
                break

        for candidate in candidates:
            if len(selected) >= desired:
                break
            if any(candidate.get("component") == item.get("component") and candidate.get("time") == item.get("time") for item in selected):
                continue
            selected.append(candidate)

        selected = sorted(selected[:desired], key=lambda item: self._parse_time_to_epoch(item.get("time", ""), query) or 0)
        return [
            {
                "component": item.get("component", ""),
                "time": item.get("time", query.start_time),
                "reason": item.get("reason", ""),
            }
            for item in selected
        ]

    def _global_root_times(
        self,
        query: RCAQuery,
        workspace: Dict[str, Any],
        topk: List[Dict[str, Any]],
        desired: int,
    ) -> List[str]:
        anomaly_details = workspace.get("association_layer", {}).get("anomaly_details", {})
        window_dur = int(query.end_ts) - int(query.start_ts)
        candidates = []
        for row in topk:
            comp = row.get("component", "")
            root_score = float(row.get("RootCauseScore", 0.0))
            for seg in anomaly_details.get(comp, []):
                seg_dur = int(seg["end_ts"]) - int(seg["start_ts"])
                if seg_dur >= window_dur * 0.75:
                    continue
                ts = self._segment_event_ts(seg)
                if ts is not None:
                    candidates.append({"timestamp": ts, "score": root_score * max(0.1, float(seg.get("severity", 0.0)))})
        if not candidates:
            old = self._global_root_time(query, workspace, topk)
            return [old] if old else []

        candidates.sort(key=lambda item: item["score"], reverse=True)
        min_gap = max(60.0, window_dur / max(desired * 2, 1))
        selected = []
        for candidate in candidates:
            if any(abs(candidate["timestamp"] - item["timestamp"]) < min_gap for item in selected):
                continue
            selected.append(candidate)
            if len(selected) >= desired:
                break
        selected = sorted(selected or candidates[:desired], key=lambda item: item["timestamp"])
        return [epoch_to_local(int(item["timestamp"])) for item in selected[:desired]]

    def _parse_time_to_epoch(self, value: str, query: RCAQuery) -> Optional[int]:
        from datetime import datetime, timedelta, timezone

        try:
            dt = datetime.strptime(str(value), "%Y-%m-%d %H:%M:%S")
            return int(dt.replace(tzinfo=timezone(timedelta(hours=8))).timestamp())
        except Exception:
            return None

    def _segment_event_ts(self, segment: Dict[str, Any]) -> Optional[int]:
        points = sorted(segment.get("points", []), key=lambda point: int(point.get("timestamp", 0)))
        if len(points) >= 2:
            peak_dev = max(float(point.get("deviation", 0.0)) for point in points)
            if peak_dev > 0 and float(points[0].get("deviation", 0.0)) >= 0.80 * peak_dev:
                return int(points[0]["timestamp"])
            best_jump = None
            for prev, cur in zip(points, points[1:]):
                jump = float(cur.get("deviation", 0.0)) - float(prev.get("deviation", 0.0))
                if jump <= 0:
                    continue
                cur_dev = float(cur.get("deviation", 0.0))
                magnitude = jump * (0.5 + 0.5 * min(1.0, cur_dev / max(peak_dev, 1e-9)))
                if cur_dev >= 0.45 * peak_dev and (best_jump is None or magnitude > best_jump[0]):
                    best_jump = (magnitude, int(cur["timestamp"]))
            if best_jump is not None and best_jump[0] > 0:
                return best_jump[1]
            peak_pt = max(points, key=lambda point: float(point.get("deviation", 0.0)))
            return int(peak_pt["timestamp"])
        if len(points) == 1:
            return int(points[0]["timestamp"])
        if segment.get("start_ts") is not None:
            return int(segment["start_ts"])
        return None

    def _global_root_time(self, query: RCAQuery, workspace: Dict[str, Any], topk: List[Dict[str, Any]]) -> Optional[str]:
        anomaly_details = workspace.get("association_layer", {}).get("anomaly_details", {})
        window_dur = int(query.end_ts) - int(query.start_ts)
        best_event = None
        for row in topk:
            comp = row.get("component", "")
            root_score = float(row.get("RootCauseScore", 0.0))
            for seg in anomaly_details.get(comp, []):
                seg_dur = int(seg["end_ts"]) - int(seg["start_ts"])
                if seg_dur >= window_dur * 0.75:
                    continue
                ts = self._segment_event_ts(seg)
                if ts is None:
                    continue
                compactness = 1.0 - min(1.0, seg_dur / max(window_dur, 1))
                score = root_score * (0.70 * float(seg.get("severity", 0.0)) + 0.30 * compactness)
                if best_event is None or score > best_event[0] or (abs(score - best_event[0]) < 1e-9 and ts < best_event[1]):
                    best_event = (score, ts)
        return epoch_to_local(int(best_event[1])) if best_event is not None else None

    def _build_explanation(self, best: Dict[str, Any]) -> str:
        return (
            f"{best['component']} is selected with FinalScore={best['FinalScore']:.3f}, "
            f"RootCauseScore={best['RootCauseScore']:.3f}, "
            f"CounterfactualScore={best['CounterfactualScore']:.3f}, "
            f"and ReasonScore={best['ReasonScore']:.3f}."
        )

    def _self_evaluate(
        self,
        result: Dict[str, Any],
        workspace: Dict[str, Any],
        params: Dict[str, Any],
    ) -> Tuple[float, List[str]]:
        final = result["final_root_cause"]
        scores = final.get("scores", {})
        quality = 0.45 * float(scores.get("FinalScore", 0.0)) + 0.30 * float(scores.get("ReasonScore", 0.0)) + 0.25
        warnings: List[str] = []
        if float(scores.get("ReasonScore", 0.0)) < 0.25:
            warnings.append("Reason evidence is weak.")
        return max(0.10, min(1.0, quality)), warnings
