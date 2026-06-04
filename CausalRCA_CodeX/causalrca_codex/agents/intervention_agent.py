from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from causalrca_codex.agents.base import BaseAgent
from causalrca_codex.core.graph_ops import max_path_strengths
from causalrca_codex.schemas import WeightedCausalGraph


class InterventionAgent(BaseAgent):
    """Agent 5: intervention ranking.

    Cross-agent contract is intentionally compact:
    - ExplainScore: how well a candidate can explain downstream anomalies.
    - RootCauseScore: intervention-stage root-cause score.

    Other quantities such as temporal precedence and source position are local
    features used only to compute RootCauseScore; they are not passed forward.
    """

    name = "InterventionAgent"
    purpose = "Rank root-cause candidates with ExplainScore and RootCauseScore"
    preconditions = ["causal_graph_layer.weighted_causal_graph", "association_layer.anomaly_scores"]
    produces = ["intervention_layer.explain_scores", "intervention_layer.root_cause_scores", "intervention_layer.ranking"]
    estimated_cost = "medium"

    def _execute(self, workspace: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        graph = workspace["causal_graph_layer"].get("weighted_causal_graph")
        anomaly_scores: Dict[str, float] = workspace["association_layer"].get("anomaly_scores", {})
        anomaly_details: Dict[str, List[Dict[str, Any]]] = workspace["association_layer"].get("anomaly_details", {})
        first_ts = workspace["association_layer"].get("first_anomaly_ts", {})
        candidate_evidence: Dict[str, float] = workspace["fault_id_layer"].get("candidate_scores", {})
        refined = list(workspace["fault_id_layer"].get("refined_candidates", []))
        tentative = workspace["fault_id_layer"].get("tentative_root_cause")

        print(f"    [InterventionAgent] graph nodes={len(graph.nodes) if graph else 0} edges={len(graph.edges) if graph else 0}")
        print(f"    [InterventionAgent] refined_candidates={len(refined)}")
        print("    [InterventionAgent] RCS = 0.40*Evidence + 0.25*Source + 0.20*Time + 0.15*ES")

        candidates = refined or list(anomaly_scores.keys())
        if tentative and tentative not in candidates:
            candidates.insert(0, tentative)

        if graph is None:
            graph = WeightedCausalGraph()
            for component in candidates:
                graph.add_node(component, severity=float(anomaly_scores.get(component, 0.0)))

        segment_first_ts: Dict[str, Optional[float]] = {}
        for comp in candidates:
            starts = [
                seg.get("start_ts")
                for seg in anomaly_details.get(comp, [])
                if seg.get("start_ts") is not None
            ]
            segment_first_ts[comp] = min(starts) if starts else first_ts.get(comp)
        t_ref = min([v for v in segment_first_ts.values() if v is not None], default=None)

        explain_scores: Dict[str, float] = {}
        root_scores: Dict[str, float] = {}

        for candidate in candidates:
            if not graph.has_node(candidate):
                graph.add_node(
                    candidate,
                    anomalous=candidate in anomaly_scores,
                    severity=float(anomaly_scores.get(candidate, 0.0)),
                )

            strengths = max_path_strengths(graph, candidate, max_depth=self.config.max_path_depth)
            explain = self._explain_score(candidate, strengths, anomaly_scores)
            explain_scores[candidate] = round(explain, 6)

            outgoing_edges = graph.outgoing(candidate)
            incoming_edges = graph.incoming(candidate)
            weighted_outgoing = sum(
                float(getattr(edge, "weight", 0.0))
                for edge in outgoing_edges
                if edge.target in anomaly_scores
            )
            total_outgoing = sum(float(getattr(edge, "weight", 0.0)) for edge in outgoing_edges)
            propagation = weighted_outgoing / total_outgoing if total_outgoing > 1e-9 else 0.0

            anomalous_incoming = [edge for edge in incoming_edges if edge.source in anomaly_scores]
            incoming_ratio = len(anomalous_incoming) / max(len(incoming_edges), 1)
            source_bonus = 1.0 if not anomalous_incoming else max(0.3, 1.0 - 0.2 * len(anomalous_incoming))
            source_score = max(0.0, min(1.0, 0.5 * propagation * (1.0 - 0.5 * incoming_ratio) + 0.5 * source_bonus))

            comp_first = segment_first_ts.get(candidate)
            if t_ref is None or comp_first is None:
                temporal_score = 0.5
            else:
                temporal_score = math.exp(-(float(comp_first) - float(t_ref)) / max(self.config.lambda_early, 1e-6))

            evidence_score = float(candidate_evidence.get(candidate, anomaly_scores.get(candidate, 0.0)))
            root_score = (
                0.40 * evidence_score
                + 0.25 * source_score
                + 0.20 * temporal_score
                + 0.15 * explain
            )
            root_scores[candidate] = round(max(0.0, min(1.0, root_score)), 6)

        ranking = [
            {
                "component": component,
                "RootCauseScore": score,
                "ExplainScore": explain_scores.get(component, 0.0),
            }
            for component, score in sorted(root_scores.items(), key=lambda item: item[1], reverse=True)
        ]

        top1 = ranking[0]["RootCauseScore"] if ranking else 0.0
        top2 = ranking[1]["RootCauseScore"] if len(ranking) > 1 else 0.0
        graph_quality = float(workspace["causal_graph_layer"].get("quality", 0.0))
        confidence = (
            self.config.delta_score * top1
            + self.config.delta_margin * max(0.0, top1 - top2)
            + self.config.delta_graph_quality * graph_quality
        )
        adaptive_k = min(
            len(ranking),
            max(self.config.top_k, int(math.ceil(math.sqrt(max(len(ranking), 1)))) + self.config.top_k),
        )
        topk = ranking[:adaptive_k]

        print(f"    [InterventionAgent] candidates={len(ranking)} top1_confidence={confidence:.4f}")
        for i, row in enumerate(ranking[:5]):
            comp = row["component"]
            evidence = float(candidate_evidence.get(comp, anomaly_scores.get(comp, 0.0)))
            print(
                f"      #{i + 1} {comp}: "
                f"RCS={row['RootCauseScore']:.4f} ES={row['ExplainScore']:.4f} Evidence={evidence:.4f}"
            )

        workspace["intervention_layer"].update(
            {
                "explain_scores": explain_scores,
                "root_cause_scores": root_scores,
                "ranking": ranking,
                "top1_confidence": round(confidence, 6),
                "topk_candidates": topk,
            }
        )
        return {
            "explain_scores": explain_scores,
            "root_cause_scores": root_scores,
            "ranking": ranking,
            "top1_confidence": round(confidence, 6),
            "topk_candidates": topk,
        }

    def _explain_score(
        self,
        candidate: str,
        strengths: Dict[str, float],
        anomaly_scores: Dict[str, float],
    ) -> float:
        explain = 0.0
        downstream_sev = 0.0
        for target, severity in anomaly_scores.items():
            if target == candidate:
                continue
            strength = strengths.get(target, 0.0)
            if strength <= 0:
                continue
            explain += float(severity) * strength
            downstream_sev += float(severity)
        if downstream_sev > 1e-9:
            return max(0.0, min(1.0, explain / downstream_sev))

        max_sev = max(anomaly_scores.values()) if anomaly_scores else 1.0
        raw = float(anomaly_scores.get(candidate, 0.0)) / max(max_sev, 1e-9)
        return max(0.0, min(1.0, raw * 0.5))

    def _self_evaluate(
        self,
        result: Dict[str, Any],
        workspace: Dict[str, Any],
        params: Dict[str, Any],
    ) -> Tuple[float, List[str]]:
        confidence = float(result["top1_confidence"])
        warnings: List[str] = []
        if not result["ranking"]:
            return 0.10, ["Intervention ranking is empty."]
        if confidence < 0.40:
            warnings.append("Top-1 intervention confidence is below 0.40.")
        if len(result["ranking"]) > 1:
            margin = result["ranking"][0]["RootCauseScore"] - result["ranking"][1]["RootCauseScore"]
            if margin < 0.05:
                warnings.append("Top candidates have very similar RootCauseScore values.")
        return max(0.10, min(1.0, confidence)), warnings
