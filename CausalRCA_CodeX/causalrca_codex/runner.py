"""Runner 模块 - 批量运行CausalRCA-Flow benchmark。

负责：
1. 读取 query.csv 和 record.csv
2. 逐行运行 OrchestratorAgent
3. 可选调用 OpenRCA 评估器
4. 输出汇总CSV和逐行诊断JSON
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from causalrca_codex.config import AgentLoopConfig
from causalrca_codex.core.dataset import build_query, dataset_path
from causalrca_codex.orchestrator import OrchestratorAgent
from causalrca_codex.utils.serialization import to_jsonable

logger = logging.getLogger("causalrca")


def _banner(msg: str, char: str = "=", width: int = 70) -> None:
    """打印醒目的分隔线标题。"""
    print(f"\n{char * width}")
    print(f"  {msg}")
    print(f"{char * width}")


def _import_evaluator(config: AgentLoopConfig):
    """动态导入 OpenRCA 评估器。"""
    root = str(config.openrca_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from main.evaluate import evaluate
        return evaluate
    except Exception:
        return None


def read_ground_truth(record_df: Optional[pd.DataFrame], idx: int) -> Optional[Dict[str, Any]]:
    """从 record.csv 读取基准答案。"""
    if record_df is None or idx >= len(record_df):
        return None
    row = record_df.iloc[idx]
    return {column: row[column] for column in record_df.columns}


def run_dataset(
    dataset: str = "Bank",
    start_idx: int = 0,
    end_idx: int = 150,
    tag: str = "causalrca-codex",
    config: Optional[AgentLoopConfig] = None,
    save_diagnostics: bool = True,
) -> pd.DataFrame:
    """批量运行CausalRCA-Flow benchmark。

    Args:
        dataset: 数据集名称 ("Bank"/"Telecom"/"Market")
        start_idx: 起始行号
        end_idx: 结束行号
        tag: 输出标签
        config: 配置对象
        save_diagnostics: 是否保存逐行诊断JSON

    Returns:
        DataFrame: 包含每行的预测、评估结果和得分
    """
    config = config or AgentLoopConfig()
    ds_dir = dataset_path(config, dataset)
    query_file = ds_dir / "query.csv"
    record_file = ds_dir / "record.csv"

    if not query_file.exists():
        raise FileNotFoundError(f"query.csv not found: {query_file}")

    query_df = pd.read_csv(query_file)
    record_df = pd.read_csv(record_file) if record_file.exists() else None

    eval_rows = []
    evaluator = _import_evaluator(config)
    out_dir = config.resolved_output_root() / Path(dataset)
    out_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = out_dir / f"{tag}-diagnostics"
    if save_diagnostics:
        diagnostics_dir.mkdir(parents=True, exist_ok=True)

    total = min(end_idx + 1, len(query_df)) - start_idx
    correct = 0
    wrong = 0

    _banner(f"CausalRCA-Flow Benchmark | {dataset} | {start_idx}~{end_idx} (共{total}条)")
    print(f"  输出目录: {out_dir}")
    print(f"  评估器: {'已加载' if evaluator else '未加载'}")
    print(f"  诊断输出: {'ON' if save_diagnostics else 'OFF'}")

    batch_start = time.time()

    for idx, row in query_df.iterrows():
        if idx < start_idx:
            continue
        if idx > end_idx:
            break

        case_start = time.time()

        _banner(f"Case {idx}/{end_idx} | Task {row['task_index']}", char="-", width=60)

        query = build_query(
            config=config,
            dataset=dataset,
            row_id=int(idx),
            task_index=str(row["task_index"]),
            instruction=str(row["instruction"]),
            scoring_points=str(row.get("scoring_points", "")),
        )
        ground_truth = read_ground_truth(record_df, idx)
        result = OrchestratorAgent(config).run(query, ground_truth=ground_truth)
        prediction_json = result["prediction_json"]

        passed = []
        failed = []
        score: Any = "N/A"
        if evaluator is not None:
            passed, failed, score = evaluator(prediction_json, query.scoring_points)

        gt_text = ""
        if ground_truth:
            gt_text = "\n".join(f"{key}: {value}" for key, value in ground_truth.items() if key != "description")

        case_elapsed = time.time() - case_start

        # 打印单case结果
        if passed and not failed:
            correct += 1
            print(f"\n  [OK] PASS | Score: {score} | 耗时: {case_elapsed:.1f}s")
        else:
            wrong += 1
            print(f"\n  [X] FAIL | Score: {score} | 耗时: {case_elapsed:.1f}s")
            if failed:
                for f in failed[:3]:
                    print(f"    失败: {f}")

        eval_rows.append(
            {
                "row_id": idx,
                "task_index": query.task_index,
                "instruction": query.instruction,
                "prediction": prediction_json,
                "groundtruth": gt_text,
                "passed": "\n".join(map(str, passed)),
                "failed": "\n".join(map(str, failed)),
                "score": score,
            }
        )

        if save_diagnostics:
            diag_file = diagnostics_dir / f"row_{idx:04d}.json"
            with diag_file.open("w", encoding="utf-8") as handle:
                json.dump(to_jsonable(result), handle, ensure_ascii=False, indent=2)

    # 汇总结果
    batch_elapsed = time.time() - batch_start
    eval_df = pd.DataFrame(eval_rows)
    out_file = out_dir / f"{tag}.csv"
    eval_df.to_csv(out_file, index=False)

    _banner("Benchmark 汇总", char="=")
    print(f"  总数: {total} | 正确: {correct} | 错误: {wrong}")
    if total > 0:
        print(f"  准确率: {correct/total*100:.1f}%")
    print(f"  总耗时: {batch_elapsed:.1f}s | 平均: {batch_elapsed/max(total,1):.1f}s/case")
    print(f"  结果文件: {out_file}")

    return eval_df
