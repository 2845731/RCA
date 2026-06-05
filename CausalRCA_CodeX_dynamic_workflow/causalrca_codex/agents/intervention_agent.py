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
        print("    [InterventionAgent] RCS = 0.35*Evidence + 0.20*Source + 0.20*Time + 0.25*Unexplained")

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

            # --- UnexplainedScore (replaces ExplainScore) ---
            # Instead of rewarding "how many downstream anomalies this node explains"
            # (which rewards hub nodes like MG01/MG02), compute "how much of this
            # node's anomaly is NOT explained by upstream propagation".
            # A node whose anomaly cannot be traced to upstream causes is more likely
            # to be the true root cause. This is the "unique grievance" concept.
            unexplained = self._unexplained_score(candidate, graph, anomaly_scores, segment_first_ts)
            explain_scores[candidate] = round(unexplained, 6)

            incoming_edges = graph.incoming(candidate)
            outgoing_edges = graph.outgoing(candidate)

            # SourceScore: penalize nodes whose parents are also anomalous
            # (they might be downstream effects, not root causes)
            anomalous_parents = [edge for edge in incoming_edges if edge.source in anomaly_scores]
            if not anomalous_parents:
                # No anomalous parents → likely a root cause (or isolated)
                source_score = 0.85
            else:
                # Has anomalous parents → less likely to be root cause
                # But if the parents' anomalies are weaker, still possible
                parent_max_sev = max(float(anomaly_scores.get(e.source, 0.0)) for e in anomalous_parents)
                my_sev = float(anomaly_scores.get(candidate, 0.0))
                # If my severity is much higher than parents, I might still be root
                sev_ratio = my_sev / max(parent_max_sev, 0.01)
                source_score = max(0.15, 0.50 * sev_ratio)

            comp_first = segment_first_ts.get(candidate)
            if t_ref is None or comp_first is None:
                temporal_score = 0.5
            else:
                temporal_score = math.exp(-(float(comp_first) - float(t_ref)) / max(self.config.lambda_early, 1e-6))

            evidence_score = float(candidate_evidence.get(candidate, anomaly_scores.get(candidate, 0.0)))
            # Formula: evidence + source + temporal + unexplained
            # Unexplained score captures "unique grievance" - anomaly not explained
            # by upstream propagation. Higher weight to reduce bias toward
            # components with many native KPIs (like MySQL).
            root_score = (
                0.35 * evidence_score
                + 0.20 * source_score
                + 0.20 * temporal_score
                + 0.25 * unexplained
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

    def _unexplained_score(
        self,
        candidate: str,
        graph: Any,
        anomaly_scores: Dict[str, float],
        segment_first_ts: Dict[str, Any],
    ) -> float:
        """Compute how much of a candidate's anomaly is NOT explained by upstream propagation.

        Key design: Don't penalize hub nodes (like Tomcat04 with many callers) just
        because they have many anomalous parents. Instead, only count parents that:
        1. Were anomalous BEFORE the candidate (temporal precedence)
        2. Are THEMSELVES likely root causes (high severity, few parents)

        A parent that is just another downstream effect (low severity, many parents)
        should not "explain away" the candidate's anomaly.
        """
        if graph is None:
            max_sev = max(anomaly_scores.values()) if anomaly_scores else 1.0
            return float(anomaly_scores.get(candidate, 0.0)) / max(max_sev, 1e-9)

        my_sev = float(anomaly_scores.get(candidate, 0.0))
        if my_sev <= 0:
            return 0.0

        my_ts = segment_first_ts.get(candidate)

        # Collect upstream (parent) contributions with quality weighting
        total_upstream_explained = 0.0
        n_quality_parents = 0

        for edge in graph.incoming(candidate):
            parent = edge.source
            parent_sev = float(anomaly_scores.get(parent, 0.0))
            if parent_sev <= 0:
                continue

            parent_ts = segment_first_ts.get(parent)
            # Only count if parent was anomalous BEFORE candidate
            if parent_ts is None or my_ts is None or parent_ts >= my_ts:
                continue

            # Discount parents that are themselves likely downstream effects:
            # If the parent has many anomalous parents, it's probably not a root cause
            # and shouldn't explain the candidate's anomaly
            parent_incoming_anomalous = sum(
                1 for e in graph.incoming(parent)
                if float(anomaly_scores.get(e.source, 0.0)) > 0
            )
            # Quality factor: parents with fewer anomalous parents are more likely
            # to be actual root causes and thus more "explaining"
            quality = 1.0 / (1.0 + 0.3 * parent_incoming_anomalous)

            # Edge weight (propagation strength)
            edge_weight = float(edge.weight)

            # Contribution: parent_sev * edge_weight * quality
            contrib = parent_sev * edge_weight * quality
            total_upstream_explained += contrib
            n_quality_parents += 1

        if n_quality_parents == 0:
            # No quality anomalous parents → anomaly is unexplained → high root cause
            return min(1.0, my_sev)

        # Fraction explained by upstream
        explained_ratio = total_upstream_explained / max(my_sev, 1e-6)
        explained_ratio = min(1.0, explained_ratio)

        # Unexplained fraction
        unexplained = 1.0 - explained_ratio

        # Scale by severity
        return max(0.0, min(1.0, unexplained * my_sev))

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
