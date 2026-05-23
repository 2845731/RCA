"""
SpectralRCA-Agent: Spectral edge validation standalone evaluation.

Evaluates the quality of spectral-validated causal edge pruning
(Innovation 2 ablation).

Usage:
    python -m experiments.run_spectral_edge --dataset Bank --date_folder 2021_03_04
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.anomaly.metric_anomaly_expert import MetricAnomalyExpert
from src.config import SpectralRCAConfig
from src.data_loader import DataLoader
from src.graph.graph_refiner import SpectralGraphRefinementExpert
from src.ranking.root_cause_ranker import RootCauseRanker


def main():
    parser = argparse.ArgumentParser(description="Spectral Edge Validation Evaluation")
    parser.add_argument("--dataset_dir", type=str, default=r"d:\GitHubDownload\OpenRCA\dataset")
    parser.add_argument("--dataset", type=str, default="Bank", choices=["Bank", "Telecom"])
    parser.add_argument("--date_folder", type=str, default=None)
    args = parser.parse_args()

    data_loader = DataLoader(args.dataset_dir)
    config = SpectralRCAConfig()

    if args.date_folder is None:
        folders = data_loader.list_date_folders(args.dataset)
        args.date_folder = folders[0] if folders else None
        if args.date_folder is None:
            print("No date folders found!")
            return

    records = data_loader.load_records(args.dataset)
    date_part = args.date_folder.replace("_", "-")
    incident_start = None
    incident_end = None
    gt_component = None

    for _, row in records.iterrows():
        dt_str = str(row.get("datetime", ""))
        if date_part in dt_str:
            ts = float(row.get("timestamp", 0))
            dt = datetime.fromtimestamp(ts)
            incident_start = (dt - timedelta(minutes=15)).strftime("%Y-%m-%d %H:%M:%S")
            incident_end = (dt + timedelta(minutes=15)).strftime("%Y-%m-%d %H:%M:%S")
            gt_component = row.get("component")
            break

    if incident_start is None:
        incident_start = f"{date_part} 00:00:00"
        incident_end = f"{date_part} 00:30:00"

    print(f"Date folder: {args.date_folder}")
    print(f"Incident window: {incident_start} to {incident_end}")
    if gt_component:
        print(f"Ground truth component: {gt_component}")

    metric_series_dict = data_loader.build_metric_series_dict(
        args.dataset, args.date_folder,
        resample_interval=config.pipeline.resample_interval,
    )

    inc_start = datetime.strptime(incident_start, "%Y-%m-%d %H:%M:%S")
    inc_end = datetime.strptime(incident_end, "%Y-%m-%d %H:%M:%S")
    duration = inc_end - inc_start
    baseline_start = (inc_start - config.pipeline.baseline_window_multiplier * duration).strftime("%Y-%m-%d %H:%M:%S")

    expert = MetricAnomalyExpert(config)
    anomaly_evidence = expert.detect(
        metric_series_dict,
        incident_start=incident_start,
        incident_end=incident_end,
        baseline_start=baseline_start,
        baseline_end=incident_start,
    )

    print(f"\nAnomaly detection found {len(anomaly_evidence)} series, "
          f"{sum(1 for e in anomaly_evidence if e.final_anomaly_score > 0.5)} anomalous")

    refiner = SpectralGraphRefinementExpert(config)
    edge_evidence = refiner.refine(
        anomaly_evidence,
        metric_series_dict,
        incident_start=incident_start,
        incident_end=incident_end,
    )

    kept_edges = [e for e in edge_evidence if e.keep_edge]
    pruned_edges = [e for e in edge_evidence if not e.keep_edge]

    print(f"\nGraph Refinement Results:")
    print(f"  Total candidate edges: {len(edge_evidence)}")
    print(f"  Kept edges: {len(kept_edges)}")
    print(f"  Pruned edges: {len(pruned_edges)}")

    print(f"\nTop kept edges (by weight):")
    for e in sorted(kept_edges, key=lambda x: x.final_edge_weight, reverse=True)[:10]:
        print(f"  {e.source} → {e.target}: weight={e.final_edge_weight:.4f}, "
              f"shape_sim={e.spectral_shape_similarity:.4f}, "
              f"freq_match={e.dominant_freq_match:.4f}, "
              f"phase_cons={e.phase_lag_consistency:.4f}")

    if gt_component:
        gt_edges = [e for e in kept_edges if gt_component in e.source]
        print(f"\nEdges from ground truth component ({gt_component}):")
        for e in gt_edges:
            print(f"  {e.source} → {e.target}: weight={e.final_edge_weight:.4f}")

    ranker = RootCauseRanker()
    ranked = ranker.rank(anomaly_evidence, edge_evidence)

    print(f"\nRoot Cause Ranking (top 5):")
    for i, c in enumerate(ranked[:5]):
        marker = " ← GT" if gt_component and gt_component in c.node_id else ""
        print(f"  {i+1}. {c.node_id}: score={c.root_score:.4f}, "
              f"anomaly={c.anomaly_score:.4f}, out={c.out_evidence:.4f}{marker}")


if __name__ == "__main__":
    main()
