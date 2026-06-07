"""Benchmark test for CausalRCA_CodeX with full logging."""
import argparse
import os
import re
import sys
import time
from datetime import datetime

from tqdm import tqdm

sys.path.insert(0, ".")
sys.path.insert(0, "CausalRCA_CodeX")
sys.path.insert(0, "..")

from causalrca_codex.config import AgentLoopConfig
from causalrca_codex.core.dataset import build_query, dataset_path
from causalrca_codex.orchestrator import OrchestratorAgent
from main.evaluate import evaluate
import pandas as pd

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


class TeeOutput:
    def __init__(self, terminal, logfile):
        self.terminal = terminal
        self.logfile = logfile

    def write(self, text):
        self.terminal.write(text)
        self.terminal.flush()
        self.logfile.write(_ANSI_RE.sub("", text))
        self.logfile.flush()

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()


def main(args):
    uid = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", uid)
    os.makedirs(output_dir, exist_ok=True)

    run_log_dir = os.path.join(output_dir, "run_logs")
    os.makedirs(run_log_dir, exist_ok=True)

    log_file = os.path.join(run_log_dir, "run.log")
    progress_file = os.path.join(run_log_dir, "progress.log")
    log_f = open(log_file, "w", encoding="utf-8")
    progress_f = open(progress_file, "w", encoding="utf-8")
    tee = TeeOutput(sys.__stdout__, log_f)
    sys.stdout = tee

    print(f"[Config] Dataset={args.dataset}, Cases={args.start_idx}~{args.end_idx}, Log={log_file}")
    print(f"  Progress : {progress_file}  (watch with: tail -f {progress_file})")
    print(f"  Output Dir: {output_dir}")

    config = AgentLoopConfig()
    ds_dir = dataset_path(config, args.dataset)
    query_df = pd.read_csv(ds_dir / "query.csv")
    record_df = pd.read_csv(ds_dir / "record.csv")

    start_idx = args.start_idx
    end_idx = min(args.end_idx, len(query_df))
    num_cases = end_idx - start_idx

    print(f"\n{'='*80}")
    print(f"  CausalRCA-CodeX Benchmark")
    print(f"{'='*80}")
    print(f"  Dataset  : {args.dataset}")
    print(f"  Cases    : {start_idx} ~ {end_idx - 1} ({num_cases} cases)")
    print(f"  Max Iters: {config.max_iterations}")
    print(f"  Output   : {output_dir}")
    print(f"{'='*80}")

    results = []
    start_time = time.time()

    # Progress bar: use tqdm when stderr is a TTY, otherwise print plain lines
    _is_tty = sys.__stderr__.isatty()
    pbar = tqdm(
        total=num_cases,
        desc=f"📊 {args.dataset}",
        unit="case",
        bar_format="{l_bar}{bar:30}{r_bar}",
        file=sys.__stderr__,
        ncols=100,
        disable=not _is_tty,
    )

    for idx in range(start_idx, end_idx):
        row = query_df.iloc[idx]
        task_index = str(row["task_index"])
        instruction = str(row["instruction"])
        scoring_points = str(row["scoring_points"])

        print(f"\n{'#'*80}")
        print(f"  Case {idx - start_idx + 1}/{num_cases} | {task_index}")
        print(f"{'#'*80}")
        print(f"  Instruction: {instruction}")
        print(f"  Scoring    : {scoring_points}")

        case_start = time.time()
        timed_out = False

        try:
            query = build_query(
                config=config,
                dataset=args.dataset,
                row_id=idx,
                task_index=task_index,
                instruction=instruction,
                scoring_points=scoring_points,
            )
            gt = dict(record_df.iloc[idx])

            orch = OrchestratorAgent(config)
            result = orch.run(query, ground_truth=gt)

            case_elapsed = time.time() - case_start
            pred_json = result["prediction_json"]
            passed, failed, score = evaluate(pred_json, query.scoring_points)

            results.append(
                {
                    "idx": idx,
                    "task": task_index,
                    "score": score,
                    "passed": len(passed),
                    "failed": len(failed),
                    "time": case_elapsed,
                }
            )

            print(f"\n  {'='*60}")
            print(f"  Case {idx} Result")
            print(f"  {'='*60}")
            print(f"  Prediction : {pred_json}")
            print(f"  Scoring    : {scoring_points}")
            print(f"  Passed     : {passed}")
            print(f"  Failed     : {failed}")
            print(f"  Score      : {score}")
            print(f"  Time       : {case_elapsed:.1f}s")
            print(f"  {'='*60}")

            # Update progress bar
            cum_score = sum(r["score"] for r in results)
            pbar.set_postfix_str(f"Score:{cum_score:.1f}/{len(results)} ({cum_score/len(results)*100:.0f}%) ⏱{case_elapsed:.0f}s")
            pbar.update(1)
            _prog_line = (
                f"[{len(results)}/{num_cases}] "
                f"Score:{cum_score:.1f} ({cum_score/len(results)*100:.0f}%) | "
                f"Case:{task_index} ✅ ⏱{case_elapsed:.0f}s"
            )
            progress_f.write(_prog_line + "\n")
            progress_f.flush()
            if not _is_tty:
                sys.__stderr__.write(_prog_line + "\n")
                sys.__stderr__.flush()

        except Exception as e:
            case_elapsed = time.time() - case_start
            import traceback

            print(f"\n  [ERROR] Case {idx} ({task_index}) failed:")
            print(f"  {traceback.format_exc()}")
            results.append(
                {
                    "idx": idx,
                    "task": task_index,
                    "score": 0.0,
                    "passed": 0,
                    "failed": 0,
                    "time": case_elapsed,
                }
            )

            # Update progress bar for error case
            cum_score = sum(r["score"] for r in results)
            pbar.set_postfix_str(f"Score:{cum_score:.1f}/{len(results)} ({cum_score/len(results)*100:.0f}%) ❌{case_elapsed:.0f}s")
            pbar.update(1)
            _prog_line = (
                f"[{len(results)}/{num_cases}] "
                f"Score:{cum_score:.1f} ({cum_score/len(results)*100:.0f}%) | "
                f"Case:{task_index} ❌ ⏱{case_elapsed:.0f}s"
            )
            progress_f.write(_prog_line + "\n")
            progress_f.flush()
            if not _is_tty:
                sys.__stderr__.write(_prog_line + "\n")
                sys.__stderr__.flush()

    elapsed = time.time() - start_time
    pbar.close()
    df = pd.DataFrame(results)

    print(f"\n{'='*80}")
    print(f"  FINAL RESULTS")
    print(f"{'='*80}")

    if not df.empty:
        total_score = df["score"].sum()
        avg_score = df["score"].mean() * 100
        total_time = df["time"].sum()

        print(f"\n+----------+----------+----------+----------+----------+")
        print(f"| Task     | Score    | Count    | Avg      | Time(s)  |")
        print(f"+----------+----------+----------+----------+----------+")
        for task, group in df.groupby("task"):
            print(
                f"| {task:<8} | {group['score'].sum():<8.2f} | {len(group):<8} "
                f"| {group['score'].mean()*100:<7.1f}% | {group['time'].sum():<8.0f} |"
            )
        print(f"+----------+----------+----------+----------+----------+")
        print(
            f"| {'TOTAL':<8} | {total_score:<8.2f} | {len(df):<8} "
            f"| {avg_score:<7.1f}% | {total_time:<8.0f} |"
        )
        print(f"+----------+----------+----------+----------+----------+")

        print(f"\n  Cumulative Score: {total_score:.1f}/{len(df)} ({avg_score:.1f}%)")
        print(f"  Total Time: {elapsed:.0f}s ({elapsed/len(df):.0f}s/case)")
    else:
        print("  No results.")

    print(f"\n[Done] All output saved to: {output_dir}")

    # Write final summary to progress file
    if not df.empty:
        total_score = df["score"].sum()
        avg_score = df["score"].mean() * 100
        progress_f.write(f"\n[DONE] Score:{total_score:.1f}/{len(df)} ({avg_score:.1f}%) | Time:{elapsed:.0f}s\n")
    else:
        progress_f.write("\n[DONE] No results.\n")
    progress_f.flush()
    progress_f.close()

    sys.stdout = sys.__stdout__
    log_f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CausalRCA-CodeX Benchmark with Logging")
    parser.add_argument("--dataset", type=str, default="Bank", help="Dataset: Bank, Telecom")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index in query.csv")
    parser.add_argument("--end_idx", type=int, default=1000, help="End index in query.csv")
    parser.add_argument("--max_iterations", type=int, default=30, help="Max iterations per case")
    args = parser.parse_args()
    main(args)
