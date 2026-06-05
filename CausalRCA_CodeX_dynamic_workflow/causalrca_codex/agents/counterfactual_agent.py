from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from causalrca_codex.agents.base import BaseAgent
from causalrca_codex.core.component import infer_component_type, normalize_component_id
from causalrca_codex.core.evidence import reason_family, semantic_match_score, signal_family
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
        print("    [CounterfactualAgent] FinalScore = 0.65*RCS + 0.35*Reason")

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
            # Simplified formula: RCS already contains causal structure information
            # (evidence + source position + temporal precedence + unexplained score).
            # CFS is unreliable because it rewards isolated nodes with no incoming edges.
            # Reason provides domain-specific confirmation.
            final_score = max(0.0, min(1.0, 0.65 * root_score + 0.35 * reason_score))

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
            # No downstream propagation: this node's anomaly doesn't explain others.
            # Penalize more heavily - a true root cause should have downstream effects.
            max_sev = max(anomaly_scores.values()) if anomaly_scores else 1.0
            raw = sev_star / max(max_sev, 1e-9)
            return max(0.0, min(1.0, raw * 0.25))

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

        # --- Pattern-based reason attribution ---
        # Instead of simple keyword matching (which is dominated by common-mode
        # OS memory signals), compute how well the OBSERVED anomaly pattern
        # matches the EXPECTED pattern for each candidate reason.
        # This prevents "high memory usage" from always winning just because
        # every component has memory metrics.
        pattern_scores = self._pattern_based_reason_scores(details, query.candidate_reasons)

        # KPI evidence: keyword-based scoring (legacy)
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
            # Use pattern-based score as primary (it's more discriminating)
            # Only fall back to keyword-based if pattern score is zero
            pattern = pattern_scores.get(reason, 0.0)
            keyword = kpi_scores.get(reason, 0.0)
            profile = float(profile_reason.get(reason, 0.0))
            # Pattern-based is always preferred when available (>0)
            # This prevents keyword/profile matching from overriding the more
            # discriminative pattern-based approach
            # When pattern=0, use a reduced version of keyword/profile to avoid
            # false matches from overly broad semantic matching
            if pattern > 0:
                kpi_evidence = pattern
            else:
                # Reduce keyword/profile influence to prevent false matches
                # (e.g., "network packet loss" getting high score from TCP-FIN-WAIT KPIs)
                kpi_evidence = max(keyword, profile) * 0.5
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

    def _pattern_based_reason_scores(
        self,
        details: List[Dict[str, Any]],
        candidate_reasons: Sequence[str],
    ) -> Dict[str, float]:
        """Compute reason scores based on anomaly pattern matching.

        For each candidate reason, define the EXPECTED anomaly signature
        (which KPI families should be deviated, in which direction).
        Then match against the OBSERVED anomaly pattern.

        Key innovation: IDF-style weighting for signal families.
        Families that appear on many KPIs (like "memory" from OS-level signals)
        are common-mode and carry less diagnostic weight. Families that are
        concentrated in a few KPIs (like "network_loss" from TCP retransmits)
        are more specific and carry more weight.
        """
        if not details:
            return {}

        # Build observed anomaly signature: which KPI families are anomalous
        observed_families: Dict[str, float] = defaultdict(float)
        family_kpi_count: Dict[str, int] = defaultdict(int)
        for seg in details:
            family = signal_family(seg.get("kpi", ""), seg.get("file_name", ""))
            severity = float(seg.get("severity", 0.0))
            observed_families[family] += severity
            family_kpi_count[family] += 1

        # IDF-style weighting: families with fewer KPIs get higher weight
        # This penalizes common-mode families (like "memory" from OS signals)
        total_kpis = sum(family_kpi_count.values()) or 1
        family_idf: Dict[str, float] = {}
        for family, count in family_kpi_count.items():
            # IDF: fewer KPIs → higher weight
            idf = math.log((1.0 + total_kpis) / (1.0 + count))
            family_idf[family] = max(0.1, idf)

        # Normalize family scores with IDF weighting
        max_family = max(observed_families.values()) if observed_families else 1.0
        observed_families = {
            f: (s / max(max_family, 1e-9)) * family_idf.get(f, 1.0)
            for f, s in observed_families.items()
        }

        # Define expected signatures for each reason
        # Each signature is a dict of {family: expected_direction_weight}
        # Positive = "high" deviation expected, Negative = "low" deviation expected
        reason_signatures = {
            "high memory usage": {
                "jvm_memory": 1.0, "memory": 0.8, "jvm_oom": 0.9,
            },
            "JVM Out of Memory (OOM) Heap": {
                "jvm_oom": 1.0, "jvm_memory": 0.9, "memory": 0.6,
            },
            "high CPU usage": {
                "cpu": 1.0, "jvm_cpu": 0.8,
            },
            "high JVM CPU load": {
                "jvm_cpu": 1.0, "cpu": 0.7,
            },
            "network packet loss": {
                "network_loss": 1.0,
            },
            "network latency": {
                "network_latency": 1.0,
            },
            "network delay": {
                "network_latency": 1.0,
            },
            "high disk I/O read usage": {
                "disk_io": 1.0, "cpu": 0.3,
            },
            "high disk space usage": {
                "disk_space": 1.0, "disk_io": 0.3,
            },
            "db connection limit": {
                "db_connection": 1.0, "db": 0.5,
            },
            "db close": {
                "db_close": 1.0, "db": 0.5,
            },
            "container memory load": {
                "jvm_memory": 0.8, "memory": 1.0,
            },
            "container CPU load": {
                "cpu": 1.0, "jvm_cpu": 0.6,
            },
            "container network packet loss": {
                "network_loss": 1.0,
            },
            "container network latency": {
                "network_latency": 1.0,
            },
            "container process termination": {
                "process": 1.0,
            },
            "node CPU load": {
                "cpu": 1.0,
            },
            "node memory consumption": {
                "memory": 1.0, "jvm_memory": 0.5,
            },
            "CPU fault": {
                "cpu": 1.0,
            },
        }

        scores: Dict[str, float] = {}
        for reason in candidate_reasons:
            sig = reason_signatures.get(reason.lower()) or reason_signatures.get(reason)
            if not sig:
                # No signature defined: fall back to family alignment
                rf = reason_family(reason)
                scores[reason] = observed_families.get(rf, 0.0)
                continue

            # Compute pattern match: how well does observed match expected
            match_score = 0.0
            n_expected = 0
            matching_kpi_count = 0
            for family, expected_weight in sig.items():
                n_expected += 1
                observed = observed_families.get(family, 0.0)
                kpi_count = family_kpi_count.get(family, 0)
                if expected_weight > 0:
                    # Expected this family to be anomalous
                    match_score += min(observed, 1.0) * expected_weight
                    matching_kpi_count += kpi_count
                else:
                    # Expected this family to be normal (low deviation)
                    match_score += (1.0 - observed) * abs(expected_weight)

            if n_expected > 0:
                match_score /= n_expected

            # Specificity bonus: reasons that match fewer, more specific KPIs
            # are more diagnostic. If a reason matches 8 memory KPIs (common-mode),
            # it's less specific than matching 1 network KPI (unique signal).
            # Use inverse of matching KPI count as specificity.
            if matching_kpi_count > 0:
                specificity = 1.0 / (1.0 + 0.15 * matching_kpi_count)
            else:
                specificity = 0.5
            match_score *= (0.6 + 0.4 * specificity)

            # Penalty: if observed has families NOT in the expected signature,
            # that's noise (e.g., memory signals when reason is "network packet loss")
            unexpected_families = set(observed_families.keys()) - set(sig.keys())
            if unexpected_families:
                unexpected_severity = sum(observed_families[f] for f in unexpected_families)
                # Mild penalty: 10% per unexpected family (capped at 30%)
                penalty = min(0.30, 0.10 * len(unexpected_families))
                match_score *= (1.0 - penalty)

            scores[reason] = max(0.0, min(1.0, match_score))

        return scores

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
                # Compactness: prefer shorter segments (they pinpoint the fault more precisely)
                compactness = 1.0 - min(1.0, seg_dur / max(window_dur, 1))
                match = semantic_match_score(reason, seg.get("kpi", ""), seg.get("file_name", "")) if reason else 0.5
                hint_match = 1.0 if reason and seg.get("reason_hint") == reason else 0.0
                severity = float(seg.get("severity", 0.0))
                # Score: prefer segments that are:
                # 1. High severity (strong anomaly)
                # 2. Compact (short duration, pinpointing the fault)
                # 3. Related to the identified reason
                # 4. NOT spanning the entire window (those are background, not fault onset)
                full_window = seg_dur >= window_dur * 0.85
                if full_window:
                    # Segments spanning most of the window are background anomalies,
                    # not the fault onset. Penalize heavily.
                    score = severity * 0.1
                else:
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
        """Find the fault onset time within an anomaly segment.

        Strategy: Find the point where deviation transitions from "baseline" to
        "anomalous". Use a higher threshold (50% of peak) to avoid picking up
        early noise. Then find the steepest rise near that threshold crossing.
        """
        points = sorted(segment.get("points", []), key=lambda point: int(point.get("timestamp", 0)))
        if not points:
            if segment.get("start_ts") is not None:
                return int(segment["start_ts"])
            return None
        if len(points) == 1:
            return int(points[0]["timestamp"])

        devs = [float(p.get("deviation", 0.0)) for p in points]
        peak_dev = max(devs)
        if peak_dev <= 0:
            return int(points[0]["timestamp"])

        # Use 50% of peak as the onset threshold
        # This is more selective than 30% and avoids picking up early noise
        onset_threshold = 0.50 * peak_dev
        onset_idx = None
        for i, d in enumerate(devs):
            if d >= onset_threshold:
                onset_idx = i
                break

        if onset_idx is None:
            # All deviations are below threshold - use peak
            peak_idx = devs.index(peak_dev)
            return int(points[peak_idx]["timestamp"])

        # Look for the steepest rise in a window around the onset
        # The steepest rise marks the actual fault trigger
        search_start = max(0, onset_idx - 2)
        search_end = min(len(devs) - 1, onset_idx + 3)
        best_rise = 0.0
        best_rise_idx = onset_idx

        for i in range(search_start, search_end):
            if i + 1 < len(devs):
                rise = devs[i + 1] - devs[i]
                if rise > best_rise:
                    best_rise = rise
                    best_rise_idx = i

        # Return the timestamp of the point just before the steepest rise
        # (the onset of the fault)
        return int(points[best_rise_idx]["timestamp"])

    def _global_root_time(self, query: RCAQuery, workspace: Dict[str, Any], topk: List[Dict[str, Any]]) -> Optional[str]:
        """Find the root cause time using multi-component convergence.

        Instead of picking the earliest spike on the top-ranked component,
        look for the time when MULTIPLE components start showing anomalies.
        This is more robust because:
        - A single component's early spike might be noise or a precursor
        - The actual fault time is when the system-wide impact begins
        - Multiple components converging at the same time = fault onset
        """
        anomaly_details = workspace.get("association_layer", {}).get("anomaly_details", {})
        window_dur = int(query.end_ts) - int(query.start_ts)
        min_gap = max(30, window_dur // 20)  # 30 seconds or 5% of window

        # Collect all segment onset times across all candidates
        onset_times: List[Dict[str, Any]] = []
        for row in topk:
            comp = row.get("component", "")
            root_score = float(row.get("RootCauseScore", 0.0))
            for seg in anomaly_details.get(comp, []):
                seg_dur = int(seg["end_ts"]) - int(seg["start_ts"])
                # Skip full-window segments (background anomalies)
                if seg_dur >= window_dur * 0.85:
                    continue
                # Skip very short segments (< 2 data points worth)
                if seg_dur < 60:
                    continue
                ts = self._segment_event_ts(seg)
                if ts is None:
                    continue
                severity = float(seg.get("severity", 0.0))
                onset_times.append({
                    "ts": ts,
                    "component": comp,
                    "severity": severity,
                    "root_score": root_score,
                    "seg_dur": seg_dur,
                })

        if not onset_times:
            return None

        # Find time clusters: multiple components with onset times close together
        # Score clusters by: diversity of component types * severity * duration
        # Prefer clusters with diverse component types (system-wide fault)
        onset_times.sort(key=lambda x: x["ts"])
        best_cluster = None

        for i, entry in enumerate(onset_times):
            cluster = [entry]
            for j, other in enumerate(onset_times):
                if i == j:
                    continue
                if abs(other["ts"] - entry["ts"]) <= min_gap * 3:
                    cluster.append(other)
            # Diversity: count unique component types (db, redis, service, middleware)
            # A fault affecting multiple types is more likely to be the root cause
            from causalrca_codex.core.component import infer_component_type
            comp_types = set(infer_component_type(e["component"]) for e in cluster)
            diversity = len(comp_types)
            unique_comps = len(set(e["component"] for e in cluster))
            avg_sev = sum(e["severity"] for e in cluster) / max(len(cluster), 1)
            # Duration bonus
            avg_dur = sum(e["seg_dur"] for e in cluster) / max(len(cluster), 1)
            dur_bonus = min(2.0, 1.0 + avg_dur / max(window_dur, 1))
            # Score: diversity is key - prefer faults affecting multiple component types
            cluster_score = diversity * unique_comps * avg_sev * dur_bonus
            if best_cluster is None or cluster_score > best_cluster[0]:
                best_cluster = (cluster_score, cluster)

        if best_cluster is None:
            # Fallback: use the highest-scoring individual onset
            best = max(onset_times, key=lambda x: x["severity"] * x["root_score"])
            return epoch_to_local(int(best["ts"]))

        # Return the median timestamp of the best cluster
        cluster_times = sorted(e["ts"] for e in best_cluster[1])
        median_ts = cluster_times[len(cluster_times) // 2]
        return epoch_to_local(int(median_ts))

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
