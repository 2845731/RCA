"""
SpectralRCA-Agent: Run full pipeline on a dataset.

Usage:
    python -m experiments.run_pipeline --dataset Bank --max_queries 5
    python -m experiments.run_pipeline --dataset Telecom --max_queries 10
"""
from __future__ import annotations

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import SpectralRCAConfig
from src.pipeline import SpectralRCAPipeline


def main():
    parser = argparse.ArgumentParser(description="SpectralRCA-Agent Pipeline")
    parser.add_argument("--dataset_dir", type=str, default=r"d:\GitHubDownload\OpenRCA\dataset",
                        help="Path to dataset directory")
    parser.add_argument("--dataset", type=str, default="Bank", choices=["Bank", "Telecom"],
                        help="Dataset to evaluate")
    parser.add_argument("--max_queries", type=int, default=None,
                        help="Maximum number of queries to process")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for results")
    args = parser.parse_args()

    config = SpectralRCAConfig()
    pipeline = SpectralRCAPipeline(
        dataset_dir=args.dataset_dir,
        config=config,
        output_dir=args.output_dir,
    )

    print(f"Running SpectralRCA-Agent on {args.dataset} dataset...")
    results = pipeline.run_dataset(args.dataset, max_queries=args.max_queries)

    print("\n" + "=" * 60)
    print(f"Results for {args.dataset}:")
    print(f"  Total queries: {results.get('total_queries', 0)}")
    print(f"  Valid queries: {results.get('valid_queries', 0)}")
    print(f"  Component Accuracy: {results.get('component_accuracy', 0.0):.4f}")
    print(f"  Reason Accuracy: {results.get('reason_accuracy', 0.0):.4f}")
    print(f"  Time Accuracy: {results.get('time_accuracy', 0.0):.4f}")
    print(f"  Top-5 Accuracy: {results.get('top5_accuracy', 0.0):.4f}")
    print(f"  Full Match Rate: {results.get('full_match_rate', 0.0):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
