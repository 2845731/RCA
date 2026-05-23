"""
SpectralRCA-Agent: Quick demo script.

Runs a single case end-to-end to demonstrate the framework.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import SpectralRCAConfig
from src.data_loader import DataLoader
from src.coordinator import Coordinator
from src.schemas import RCAQuery
from datetime import datetime


def main():
    dataset_dir = r"d:\GitHubDownload\OpenRCA\dataset"
    dataset = "Bank"

    config = SpectralRCAConfig()
    data_loader = DataLoader(dataset_dir)
    coordinator = Coordinator(config, data_loader)

    queries = data_loader.build_queries(dataset)
    if not queries:
        print("No queries found!")
        return

    query = queries[0]
    print(f"Running SpectralRCA-Agent on query: {query.task_index}")
    print(f"  Instruction: {query.instruction[:80]}...")
    print(f"  Time window: {query.start_time} to {query.end_time}")

    result = coordinator.run(query)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Prediction: {result.prediction_json}")
    print(f"\nTop-5 Root Cause Candidates:")
    for i, cand in enumerate(result.ranked_candidates[:5]):
        node_id = cand.get("node_id", "unknown")
        score = cand.get("root_score", 0.0)
        print(f"  {i+1}. {node_id}: score={score:.4f}")

    print(f"\nAnomaly Evidence: {len(result.anomaly_evidence)} series analyzed")
    high_conf = sum(1 for e in result.anomaly_evidence if e.final_anomaly_score > 0.7)
    print(f"  High confidence anomalies: {high_conf}")

    print(f"\nEdge Evidence: {len(result.edge_evidence)} edges analyzed")
    kept = sum(1 for e in result.edge_evidence if e.keep_edge)
    print(f"  Kept edges: {kept}")

    print(f"\nReasoning Trajectory: {len(result.trajectory)} steps")
    for step in result.trajectory[:5]:
        print(f"  [{step.get('state', step.get('step', '?'))}] {step.get('description', '')}")

    gt = data_loader.get_ground_truth(query.dataset, query.start_ts, query.end_ts)
    if gt:
        print(f"\nGround Truth: {gt[0].get('component', 'N/A')} - {gt[0].get('reason', 'N/A')}")

    print("\nDemo complete!")


if __name__ == "__main__":
    main()
