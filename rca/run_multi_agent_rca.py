from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main.evaluate import evaluate
from rca.multi_agent_rca import Coordinator, CoordinatorConfig
from rca.multi_agent_rca.core.dataset import build_query
from rca.multi_agent_rca.core.io_utils import append_jsonl
from rca.multi_agent_rca.evaluation import element_accuracy


def run_dataset(args: argparse.Namespace, dataset: str, run_id: str) -> Path:
    query_path = PROJECT_ROOT / "dataset" / Path(dataset) / "query.csv"
    if not query_path.exists():
        raise FileNotFoundError(f"Missing query file: {query_path}")

    out_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "test" / "multi_agent_rca" / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_dataset = dataset.replace("/", "_")
    output_path = Path(args.output) if args.output else out_dir / f"multi-agent-rca-{safe_dataset}-{args.tag}-{run_id}.csv"
    trace_path = out_dir / f"multi-agent-rca-{safe_dataset}-{args.tag}-{run_id}.jsonl"

    config = CoordinatorConfig(
        enable_sampler=not args.no_sampler,
        enable_trace=not args.no_trace,
        enable_log=not args.no_log,
        enable_verifier=not args.no_verifier,
        enable_memory=args.use_memory,
        enable_meta=not args.no_meta,
        enable_recheck_debate=not args.no_recheck_debate,
        sampler_max_traces=args.sampler_max_traces,
        sampler_max_log_templates=args.sampler_max_log_templates,
        frozen_memory=not args.update_memory,
        memory_dir=args.memory_dir,
    )
    coordinator = Coordinator(config)
    df = pd.read_csv(query_path)
    rows: List[Dict[str, Any]] = []

    end_idx = min(args.end_idx, len(df) - 1) if args.end_idx is not None else len(df) - 1
    for idx, row in df.iterrows():
        if idx < args.start_idx:
            continue
        if idx > end_idx:
            break
        task_index = str(row.get("task_index", ""))
        instruction = str(row.get("instruction", ""))
        scoring_points = str(row.get("scoring_points", ""))
        try:
            query = build_query(dataset, idx, task_index, instruction)
            result = coordinator.run(query)
            passed, failed, score = evaluate(result.prediction_json, scoring_points)
            elements = element_accuracy(result.prediction_json, scoring_points)
            coordinator.record_memory(query, result, score)
            out_row = {
                "instruction": instruction,
                "prediction": result.prediction_json,
                "groundtruth": scoring_points,
                "passed": "\n".join(passed),
                "failed": "\n".join(failed),
                "score": score,
                "row_id": idx,
                "task_index": task_index,
                **elements,
                "elapsed_seconds": result.cost.get("elapsed_seconds", 0.0),
                "llm_calls": result.cost.get("llm_calls", 0),
            }
            rows.append(out_row)
            append_jsonl(
                trace_path,
                {
                    "row_id": idx,
                    "task_index": task_index,
                    "score": score,
                    "prediction": result.prediction_json,
                    "ranked_candidates": result.ranked_candidates,
                    "cost": result.cost,
                    "diagnostics": result.diagnostics,
                    "trajectory": result.trajectory,
                    "evidence_chain": result.evidence_chain,
                },
            )
            print(f"[{dataset} #{idx}] score={score} prediction={result.prediction_json.replace(os.linesep, ' ')}")
        except Exception as exc:
            rows.append(
                {
                    "instruction": instruction,
                    "prediction": "{}",
                    "groundtruth": scoring_points,
                    "passed": "",
                    "failed": scoring_points,
                    "score": 0.0,
                    "row_id": idx,
                    "task_index": task_index,
                    "error": str(exc),
                }
            )
            append_jsonl(trace_path, {"row_id": idx, "task_index": task_index, "error": str(exc)})
            print(f"[{dataset} #{idx}] ERROR {exc}")

        pd.DataFrame(rows).to_csv(output_path, index=False)

    print(f"Saved predictions to {output_path}")
    print(f"Saved diagnostics to {trace_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-agent EvidenceGraph RCA on OpenRCA datasets.")
    parser.add_argument("--dataset", default="Bank", help="Dataset name, e.g. Bank or Telecom. Use --auto for both.")
    parser.add_argument("--auto", action="store_true", help="Run Bank and Telecom if available.")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--tag", default="prototype")
    parser.add_argument("--output", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--memory_dir", default=None)
    parser.add_argument("--use_memory", action="store_true")
    parser.add_argument("--update_memory", action="store_true")
    parser.add_argument("--no_sampler", action="store_true")
    parser.add_argument("--no_trace", action="store_true")
    parser.add_argument("--no_log", action="store_true")
    parser.add_argument("--no_verifier", action="store_true")
    parser.add_argument("--no_meta", action="store_true")
    parser.add_argument("--no_recheck_debate", action="store_true")
    parser.add_argument("--sampler_max_traces", type=int, default=160)
    parser.add_argument("--sampler_max_log_templates", type=int, default=80)
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    datasets = ["Bank", "Telecom"] if args.auto else [args.dataset]
    for dataset in datasets:
        run_dataset(args, dataset, run_id)


if __name__ == "__main__":
    main()
