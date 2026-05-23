from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

from rca.multi_agent_rca.core.schema import EvidenceGraph, EvidenceReport, RCAQuery
from rca.multi_agent_rca.meta.meta_causal_graph import MetaCausalGraph


class CausalAgent:
    """Fuse multi-agent evidence into an EvidenceGraph and ranked RCA candidates."""

    def __init__(self, use_meta: bool = True) -> None:
        self.use_meta = use_meta

    def run(
        self,
        query: RCAQuery,
        metric_report: EvidenceReport,
        trace_report: Optional[EvidenceReport],
        log_report: Optional[EvidenceReport],
        memory_cases: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[EvidenceGraph, List[Dict[str, Any]]]:
        graph = EvidenceGraph()
        component_scores: Dict[str, float] = defaultdict(float)
        reason_scores: Dict[str, float] = defaultdict(float)
        component_reason_scores: Dict[Tuple[str, str], float] = defaultdict(float)
        component_segments: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        meta = MetaCausalGraph.default(query.candidate_reasons)
        metric_segments = list(metric_report.details.get("segments", []))
        for seg in metric_segments:
            comp = str(seg.get("component", ""))
            kpi = str(seg.get("kpi", ""))
            severity = float(seg.get("severity", 0.0) or 0.0)
            reason = seg.get("reason_hint")
            component_scores[comp] += severity
            component_segments[comp].append(seg)
            graph.add_node(f"component:{comp}", "component", comp, severity, source="metric")
            graph.add_node(f"metric:{comp}:{kpi}", "metric", kpi, severity, component=comp)
            graph.add_edge(f"metric:{comp}:{kpi}", f"component:{comp}", "observed_on", severity)
            if reason:
                reason_scores[str(reason)] += severity
                component_reason_scores[(comp, str(reason))] += severity
                graph.add_node(f"reason:{reason}", "reason", str(reason), severity, source="metric_hint")
                graph.add_edge(f"metric:{comp}:{kpi}", f"reason:{reason}", "supports_reason", severity)
            if self.use_meta:
                for _, reason_node, weight in meta.instantiate_edges(comp, kpi, severity):
                    reason_name = reason_node.split("reason:", 1)[1]
                    reason_scores[reason_name] += weight
                    component_reason_scores[(comp, reason_name)] += weight
                    graph.add_node(reason_node, "reason", reason_name, weight, source="meta_causal")
                    graph.add_edge(f"metric:{comp}:{kpi}", reason_node, "meta_supports_reason", weight)

        if trace_report:
            for comp, score in trace_report.details.get("component_scores", {}).items():
                component_scores[comp] += float(score) * 1.2
                graph.add_node(f"component:{comp}", "component", comp, float(score), source="trace")
            for edge in trace_report.details.get("top_edges", []):
                src = edge.get("source")
                dst = edge.get("target")
                count = float(edge.get("count", 0.0) or 0.0)
                graph.add_node(f"component:{src}", "component", str(src), 0.0)
                graph.add_node(f"component:{dst}", "component", str(dst), 0.0)
                graph.add_edge(f"component:{src}", f"component:{dst}", "calls", min(1.0, count / 50.0), count=count)

        if log_report:
            for comp, score in log_report.details.get("component_scores", {}).items():
                component_scores[comp] += float(score)
                graph.add_node(f"component:{comp}", "component", comp, float(score), source="log")
            for reason, score in log_report.details.get("reason_scores", {}).items():
                reason_scores[reason] += float(score)
                graph.add_node(f"reason:{reason}", "reason", reason, float(score), source="log")

        for case in memory_cases or []:
            for comp in case.get("components", []):
                if comp in query.candidate_components:
                    component_scores[comp] += 0.5
                    graph.add_node(f"component:{comp}", "component", comp, 0.5, source="memory")
            for reason in case.get("reasons", []):
                if reason in query.candidate_reasons:
                    reason_scores[reason] += 0.5
                    graph.add_node(f"reason:{reason}", "reason", reason, 0.5, source="memory")

        ranked_components = sorted(component_scores.items(), key=lambda x: x[1], reverse=True)
        ranked_reasons = sorted(reason_scores.items(), key=lambda x: x[1], reverse=True)

        candidates: List[Dict[str, Any]] = []
        used_components = set()
        count = max(1, query.failure_count)
        if self._is_time_only_query(query):
            for seg in self._rank_time_segments(metric_segments):
                comp = str(seg.get("component", ""))
                if comp in used_components:
                    continue
                reason = self._best_reason_for_component(
                    comp, component_reason_scores, ranked_reasons, query.candidate_reasons
                )
                candidates.append(
                    {
                        "component": comp,
                        "reason": reason,
                        "occurrence_time": seg.get("start_time", query.start_time),
                        "occurrence_ts": seg.get("start_ts", query.start_ts),
                        "score": round(self._time_segment_score(seg), 4),
                        "support": {
                            "metric_segment": seg,
                            "component_score": round(float(component_scores.get(comp, 0.0)), 4),
                            "reason_score": round(float(reason_scores.get(reason, 0.0)), 4),
                            "selection_mode": "time_only_metric_segment",
                        },
                    }
                )
                used_components.add(comp)
                if len(candidates) >= count:
                    break

        if not candidates:
            for comp, comp_score in ranked_components[: max(count * 3, 3)]:
                if comp in used_components:
                    continue
                seg = self._best_segment(component_segments.get(comp, []))
                reason = self._best_reason_for_component(comp, component_reason_scores, ranked_reasons, query.candidate_reasons)
                total_score = comp_score + float(component_reason_scores.get((comp, reason), 0.0))
                candidates.append(
                    {
                        "component": comp,
                        "reason": reason,
                        "occurrence_time": seg.get("start_time") if seg else query.start_time,
                        "occurrence_ts": seg.get("start_ts") if seg else query.start_ts,
                        "score": round(total_score, 4),
                        "support": {
                            "metric_segment": seg,
                            "component_score": round(comp_score, 4),
                            "reason_score": round(float(reason_scores.get(reason, 0.0)), 4),
                        },
                    }
                )
                used_components.add(comp)
                if len(candidates) >= count:
                    break

        if not candidates:
            fallback_reason = ranked_reasons[0][0] if ranked_reasons else (query.candidate_reasons[0] if query.candidate_reasons else "")
            fallback_component = query.candidate_components[0] if query.candidate_components else "unknown"
            candidates.append(
                {
                    "component": fallback_component,
                    "reason": fallback_reason,
                    "occurrence_time": query.start_time,
                    "occurrence_ts": query.start_ts,
                    "score": 0.0,
                    "support": {"fallback": True},
                }
            )

        graph.provenance.extend(
            [
                metric_report.agent_name,
                trace_report.agent_name if trace_report else "TraceAgent(disabled)",
                log_report.agent_name if log_report else "LogAgent(disabled)",
                "MetaCausalGraph" if self.use_meta else "MetaCausalGraph(disabled)",
            ]
        )
        return graph, candidates

    def to_prediction_json(self, query: RCAQuery, candidates: List[Dict[str, Any]]) -> str:
        result: Dict[str, Dict[str, str]] = {}
        for idx, cand in enumerate(candidates[: query.failure_count], start=1):
            item: Dict[str, str] = {}
            if "root cause occurrence datetime" in query.target_fields:
                item["root cause occurrence datetime"] = str(cand.get("occurrence_time", query.start_time))
            if "root cause component" in query.target_fields:
                item["root cause component"] = str(cand.get("component", ""))
            if "root cause reason" in query.target_fields:
                item["root cause reason"] = str(cand.get("reason", ""))
            result[str(idx)] = item
        return json.dumps(result, ensure_ascii=False, indent=4)

    def _best_segment(self, segments: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not segments:
            return None
        return sorted(segments, key=lambda s: float(s.get("severity", 0.0) or 0.0), reverse=True)[0]

    def _is_time_only_query(self, query: RCAQuery) -> bool:
        return list(query.target_fields) == ["root cause occurrence datetime"]

    def _rank_time_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return sorted(segments, key=self._time_segment_score, reverse=True)

    def _time_segment_score(self, segment: Dict[str, Any]) -> float:
        severity = float(segment.get("severity", 0.0) or 0.0)
        start_ts = int(segment.get("start_ts", 0) or 0)
        end_ts = int(segment.get("end_ts", start_ts) or start_ts)
        duration_bonus = min(3.0, max(0, end_ts - start_ts) / 180.0)
        reason_bonus = 1.5 if segment.get("reason_hint") else 0.0
        return severity + duration_bonus + reason_bonus

    def _best_reason_for_component(
        self,
        component: str,
        component_reason_scores: Dict[Tuple[str, str], float],
        ranked_reasons: List[Tuple[str, float]],
        candidate_reasons: Iterable[str],
    ) -> str:
        local = [
            (reason, score)
            for (comp, reason), score in component_reason_scores.items()
            if comp == component
        ]
        if local:
            return sorted(local, key=lambda x: x[1], reverse=True)[0][0]
        if ranked_reasons:
            return ranked_reasons[0][0]
        candidates = list(candidate_reasons)
        return candidates[0] if candidates else ""
