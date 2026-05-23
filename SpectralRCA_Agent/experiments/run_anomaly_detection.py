"""
SpectralRCA-Agent: Standalone anomaly detection evaluation.

This script runs ONLY the anomaly detection module independently,
allowing comparison between traditional methods and spectral-enhanced methods.

Usage:
    python -m experiments.run_anomaly_detection --dataset Bank --date_folder 2021_03_04
    python -m experiments.run_anomaly_detection --dataset Bank --compare
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.anomaly.metric_anomaly_expert import MetricAnomalyExpert
from src.anomaly.traditional import traditional_anomaly_score, robust_z_scores
from src.anomaly.spectral import spectral_anomaly_score, classify_spectral_anomaly, compute_spectral_features
from src.config import AnomalyConfig, SpectralRCAConfig
from src.data_loader import DataLoader


def run_single_date(
    dataset_dir: str,
    dataset: str,
    date_folder: str,
    incident_start: str,
    incident_end: str,
    config: SpectralRCAConfig,
) -> Dict[str, Any]:
    """Run anomaly detection on a single date folder."""
    data_loader = DataLoader(dataset_dir)
    metric_series_dict = data_loader.build_metric_series_dict(
        dataset, date_folder,
        resample_interval=config.pipeline.resample_interval,
    )

    expert = MetricAnomalyExpert(config)

    baseline_multiplier = config.pipeline.baseline_window_multiplier
    inc_start = datetime.strptime(incident_start, "%Y-%m-%d %H:%M:%S")
    inc_end = datetime.strptime(incident_end, "%Y-%m-%d %H:%M:%S")
    duration = inc_end - inc_start
    baseline_start = (inc_start - baseline_multiplier * duration).strftime("%Y-%m-%d %H:%M:%S")

    anomaly_evidence = expert.detect(
        metric_series_dict,
        incident_start=incident_start,
        incident_end=incident_end,
        baseline_start=baseline_start,
        baseline_end=incident_start,
    )

    return {
        "date_folder": date_folder,
        "incident_start": incident_start,
        "incident_end": incident_end,
        "total_series": len(metric_series_dict),
        "anomaly_evidence": [ev.to_dict() for ev in anomaly_evidence],
    }


def compare_traditional_vs_spectral(
    dataset_dir: str,
    dataset: str,
) -> Dict[str, Any]:
    """Compare traditional-only vs spectral-enhanced anomaly detection.

    This is the key ablation experiment for Innovation 1:
    showing that spectral features improve anomaly detection quality.
    """
    data_loader = DataLoader(dataset_dir)
    records_df = data_loader.load_records(dataset)
    queries = data_loader.build_queries(dataset)

    config_traditional = SpectralRCAConfig()
    config_traditional.enable_spectral_anomaly = False
    config_traditional.enable_abductive = False
    config_traditional.enable_memory = False

    config_spectral = SpectralRCAConfig()
    config_spectral.enable_spectral_anomaly = True
    config_spectral.enable_abductive = False
    config_spectral.enable_memory = False

    expert_traditional = MetricAnomalyExpert(config_traditional)
    expert_spectral = MetricAnomalyExpert(config_spectral)

    comparison_results: List[Dict[str, Any]] = []

    for query in queries:
        from src.data_loader import _parse_date_folder
        date_folder = _parse_date_folder(query.instruction)
        if date_folder is None:
            continue

        metric_series_dict = data_loader.build_metric_series_dict(
            dataset, date_folder,
            resample_interval="60s",
        )
        if not metric_series_dict:
            continue

        baseline_start = _compute_baseline_start(query, config_spectral)

        trad_evidence = expert_traditional.detect(
            metric_series_dict,
            incident_start=query.start_time,
            incident_end=query.end_time,
            baseline_start=baseline_start,
            baseline_end=query.start_time,
        )

        spec_evidence = expert_spectral.detect(
            metric_series_dict,
            incident_start=query.start_time,
            incident_end=query.end_time,
            baseline_start=baseline_start,
            baseline_end=query.start_time,
        )

        ground_truth = data_loader.get_ground_truth(query.dataset, query.start_ts, query.end_ts)
        gt_component = ground_truth[0].get("component") if ground_truth else None

        trad_detected = [e for e in trad_evidence if e.final_anomaly_score > 0.5]
        spec_detected = [e for e in spec_evidence if e.final_anomaly_score > 0.5]

        trad_hits_gt = any(gt_component and gt_component in e.node_id for e in trad_detected) if gt_component else False
        spec_hits_gt = any(gt_component and gt_component in e.node_id for e in spec_detected) if gt_component else False

        trad_precision = _compute_precision(trad_detected, gt_component)
        spec_precision = _compute_precision(spec_detected, gt_component)

        comparison_results.append({
            "task_index": query.task_index,
            "gt_component": gt_component,
            "traditional_detected": len(trad_detected),
            "spectral_detected": len(spec_detected),
            "traditional_hits_gt": trad_hits_gt,
            "spectral_hits_gt": spec_hits_gt,
            "traditional_precision": trad_precision,
            "spectral_precision": spec_precision,
            "traditional_top_score": max((e.final_anomaly_score for e in trad_evidence), default=0.0),
            "spectral_top_score": max((e.final_anomaly_score for e in spec_evidence), default=0.0),
        })

    total = len(comparison_results)
    trad_recall = sum(1 for r in comparison_results if r["traditional_hits_gt"]) / max(total, 1)
    spec_recall = sum(1 for r in comparison_results if r["spectral_hits_gt"]) / max(total, 1)
    trad_prec = np.mean([r["traditional_precision"] for r in comparison_results]) if comparison_results else 0.0
    spec_prec = np.mean([r["spectral_precision"] for r in comparison_results]) if comparison_results else 0.0

    return {
        "total_queries": total,
        "traditional_recall": round(trad_recall, 4),
        "spectral_recall": round(spec_recall, 4),
        "traditional_avg_precision": round(trad_prec, 4),
        "spectral_avg_precision": round(spec_prec, 4),
        "improvement_recall": round(spec_recall - trad_recall, 4),
        "improvement_precision": round(spec_prec - trad_prec, 4),
        "per_query": comparison_results,
    }


def _compute_precision(detected, gt_component: str) -> float:
    """Compute precision: fraction of detected anomalies that are the ground truth."""
    if not detected or not gt_component:
        return 0.0
    hits = sum(1 for e in detected if gt_component in e.node_id)
    return hits / len(detected)


def _compute_baseline_start(query, config: SpectralRCAConfig) -> str:
    """Compute baseline start time from query."""
    inc_start = datetime.strptime(query.start_time, "%Y-%m-%d %H:%M:%S")
    inc_end = datetime.strptime(query.end_time, "%Y-%m-%d %H:%M:%S")
    duration = inc_end - inc_start
    baseline_start = inc_start - config.pipeline.baseline_window_multiplier * duration
    return baseline_start.strftime("%Y-%m-%d %H:%M:%S")


def main():
    parser = argparse.ArgumentParser(description="SpectralRCA Anomaly Detection Evaluation")
    parser.add_argument("--dataset_dir", type=str, default=r"d:\GitHubDownload\OpenRCA\dataset")
    parser.add_argument("--dataset", type=str, default="Bank", choices=["Bank", "Telecom"])
    parser.add_argument("--date_folder", type=str, default=None)
    parser.add_argument("--compare", action="store_true",
                        help="Compare traditional vs spectral methods")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.compare:
        print("Comparing traditional vs spectral-enhanced anomaly detection...")
        results = compare_traditional_vs_spectral(args.dataset_dir, args.dataset)
        print("\n" + "=" * 60)
        print("Anomaly Detection Comparison Results:")
        print(f"  Traditional Recall:    {results['traditional_recall']:.4f}")
        print(f"  Spectral Recall:       {results['spectral_recall']:.4f}")
        print(f"  Recall Improvement:    {results['improvement_recall']:+.4f}")
        print(f"  Traditional Precision: {results['traditional_avg_precision']:.4f}")
        print(f"  Spectral Precision:    {results['spectral_avg_precision']:.4f}")
        print(f"  Precision Improvement: {results['improvement_precision']:+.4f}")
        print("=" * 60)
    else:
        if args.date_folder is None:
            data_loader = DataLoader(args.dataset_dir)
            folders = data_loader.list_date_folders(args.dataset)
            if folders:
                args.date_folder = folders[0]
                print(f"Using first available date folder: {args.date_folder}")
            else:
                print("No date folders found!")
                return

        inc_start, inc_end = _infer_incident_window(args.dataset_dir, args.dataset, args.date_folder)
        config = SpectralRCAConfig()
        results = run_single_date(
            args.dataset_dir, args.dataset, args.date_folder,
            inc_start, inc_end, config,
        )

        print(f"\nAnomaly Detection Results for {args.date_folder}:")
        print(f"  Total series: {results['total_series']}")
        detected = [e for e in results['anomaly_evidence'] if e['final_anomaly_score'] > 0.5]
        print(f"  Detected anomalies: {len(detected)}")
        for e in sorted(detected, key=lambda x: x['final_anomaly_score'], reverse=True)[:10]:
            print(f"    {e['node_id']}: score={e['final_anomaly_score']:.4f}, "
                  f"type={e['anomaly_type']}, trad={e['traditional_score']:.4f}, "
                  f"spectral={e['spectral_score']:.4f}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


def _infer_incident_window(dataset_dir: str, dataset: str, date_folder: str) -> tuple:
    """Infer incident window from ground truth records."""
    data_loader = DataLoader(dataset_dir)
    records = data_loader.load_records(dataset)

    date_part = date_folder.replace("_", "-")
    for _, row in records.iterrows():
        dt_str = str(row.get("datetime", ""))
        if date_part in dt_str:
            ts = float(row.get("timestamp", 0))
            dt = datetime.fromtimestamp(ts)
            start = (dt - timedelta(minutes=15)).strftime("%Y-%m-%d %H:%M:%S")
            end = (dt + timedelta(minutes=15)).strftime("%Y-%m-%d %H:%M:%S")
            return start, end

    return f"{date_part} 00:00:00", f"{date_part} 00:30:00"


if __name__ == "__main__":
    main()
