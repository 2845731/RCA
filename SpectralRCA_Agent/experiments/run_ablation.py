"""
SpectralRCA-Agent: Ablation study.

Runs the full pipeline with different features disabled to measure
the contribution of each innovation.

Ablation configurations:
    - full: All innovations enabled (default)
    - no_spectral: Disable spectral anomaly detection (Innovation 1 ablation)
    - no_abductive: Disable abductive reasoning (Innovation 1 ablation)
    - no_memory: Disable self-evolving memory (Innovation 2 ablation)
    - no_graph_spectral: Disable spectral edge validation (Innovation 2 ablation)
    - traditional_only: Baseline with only traditional methods

Usage:
    python -m experiments.run_ablation --dataset Bank --ablation full
    python -m experiments.run_ablation --dataset Bank --ablation all
    python -m experiments.run_ablation --dataset Bank --ablation no_spectral,no_abductive
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import SpectralRCAConfig
from src.pipeline import SpectralRCAPipeline


ABLATION_CONFIGS = {
    "full": "All innovations enabled",
    "no_spectral": "Disable spectral anomaly detection",
    "no_abductive": "Disable abductive reasoning",
    "no_memory": "Disable self-evolving memory",
    "no_graph_spectral": "Disable spectral edge validation",
    "traditional_only": "Baseline: traditional methods only",
}


def run_single_ablation(
    dataset_dir: str,
    dataset: str,
    ablation: str,
    max_queries: int = None,
    output_dir: str = None,
) -> Dict[str, Any]:
    """Run a single ablation experiment."""
    config = SpectralRCAConfig()

    if ablation == "no_spectral":
        config.enable_spectral_anomaly = False
    elif ablation == "no_abductive":
        config.enable_abductive = False
    elif ablation == "no_memory":
        config.enable_memory = False
    elif ablation == "no_graph_spectral":
        config.enable_spectral_shape = False
        config.enable_dominant_freq_match = False
        config.enable_phase_lag = False
    elif ablation == "traditional_only":
        config.enable_spectral_anomaly = False
        config.enable_abductive = False
        config.enable_memory = False
        config.enable_spectral_shape = False
        config.enable_dominant_freq_match = False
        config.enable_phase_lag = False
    elif ablation != "full":
        raise ValueError(f"Unknown ablation: {ablation}")

    pipeline = SpectralRCAPipeline(
        dataset_dir=dataset_dir,
        config=config,
        output_dir=output_dir,
    )

    print(f"\nRunning ablation: {ablation} ({ABLATION_CONFIGS.get(ablation, '')})")
    results = pipeline.run_dataset(dataset, max_queries=max_queries)
    results["ablation_config"] = ablation
    results["ablation_description"] = ABLATION_CONFIGS.get(ablation, "")
    return results


def run_all_ablations(
    dataset_dir: str,
    dataset: str,
    max_queries: int = None,
    output_dir: str = None,
) -> Dict[str, Any]:
    """Run all ablation experiments and compare results."""
    all_results = {}
    summary_rows = []

    for ablation_name in ABLATION_CONFIGS:
        results = run_single_ablation(
            dataset_dir, dataset, ablation_name,
            max_queries=max_queries, output_dir=output_dir,
        )
        all_results[ablation_name] = results

        summary_rows.append({
            "ablation": ablation_name,
            "description": ABLATION_CONFIGS[ablation_name],
            "component_accuracy": results.get("component_accuracy", 0.0),
            "reason_accuracy": results.get("reason_accuracy", 0.0),
            "time_accuracy": results.get("time_accuracy", 0.0),
            "top5_accuracy": results.get("top5_accuracy", 0.0),
            "full_match_rate": results.get("full_match_rate", 0.0),
        })

    comparison = {
        "dataset": dataset,
        "max_queries": max_queries,
        "ablation_summary": summary_rows,
        "detailed_results": all_results,
    }

    return comparison


def main():
    parser = argparse.ArgumentParser(description="SpectralRCA-Agent Ablation Study")
    parser.add_argument("--dataset_dir", type=str, default=r"d:\GitHubDownload\OpenRCA\dataset")
    parser.add_argument("--dataset", type=str, default="Bank", choices=["Bank", "Telecom"])
    parser.add_argument("--ablation", type=str, default="full",
                        help="Ablation config name or 'all' to run all")
    parser.add_argument("--max_queries", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.ablation == "all":
        results = run_all_ablations(
            args.dataset_dir, args.dataset,
            max_queries=args.max_queries,
        )

        print("\n" + "=" * 80)
        print("ABLATION STUDY COMPARISON")
        print("=" * 80)
        print(f"{'Ablation':<25} {'Comp Acc':>10} {'Reason Acc':>10} {'Time Acc':>10} {'Top-5':>10} {'Full':>10}")
        print("-" * 80)
        for row in results["ablation_summary"]:
            print(f"{row['ablation']:<25} {row['component_accuracy']:>10.4f} "
                  f"{row['reason_accuracy']:>10.4f} {row['time_accuracy']:>10.4f} "
                  f"{row['top5_accuracy']:>10.4f} {row['full_match_rate']:>10.4f}")
        print("=" * 80)
    else:
        ablations = [a.strip() for a in args.ablation.split(",")]
        if len(ablations) == 1:
            results = run_single_ablation(
                args.dataset_dir, args.dataset, ablations[0],
                max_queries=args.max_queries,
            )
        else:
            results = {}
            for abl in ablations:
                results[abl] = run_single_ablation(
                    args.dataset_dir, args.dataset, abl,
                    max_queries=args.max_queries,
                )

        print("\n" + "=" * 60)
        if isinstance(results, dict) and "ablation_config" in results:
            print(f"Ablation: {results['ablation_config']}")
            print(f"  Component Accuracy: {results.get('component_accuracy', 0.0):.4f}")
            print(f"  Reason Accuracy:    {results.get('reason_accuracy', 0.0):.4f}")
            print(f"  Time Accuracy:      {results.get('time_accuracy', 0.0):.4f}")
            print(f"  Top-5 Accuracy:     {results.get('top5_accuracy', 0.0):.4f}")
            print(f"  Full Match Rate:    {results.get('full_match_rate', 0.0):.4f}")
        print("=" * 60)

    if args.output:
        def make_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(v) for v in obj]
            elif isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            elif isinstance(obj, tuple):
                return list(obj)
            else:
                return str(obj)

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(make_serializable(results), f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
