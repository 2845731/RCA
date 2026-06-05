"""CLI 模块 - CausalRCA-Flow 命令行入口。

用法：
    python run_causalrca_codex.py --dataset Bank --start_idx 0 --end_idx 10
    python run_causalrca_codex.py --dataset Telecom --tag my-experiment
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import AgentLoopConfig
from .runner import run_dataset


def setup_logging(level: str = "INFO") -> None:
    """配置日志系统。"""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


def main() -> None:
    """CLI主入口。"""
    parser = argparse.ArgumentParser(
        description="CausalRCA-Flow: 基于因果推理的微服务根因分析多Agent系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_causalrca_codex.py --dataset Bank --start_idx 0 --end_idx 10
  python run_causalrca_codex.py --dataset Telecom --tag experiment-1
  python run_causalrca_codex.py --dataset Bank --no_diagnostics
        """,
    )
    parser.add_argument("--dataset", type=str, default="Bank", help="数据集名称 (Bank/Telecom/Market)")
    parser.add_argument("--start_idx", type=int, default=0, help="起始行号")
    parser.add_argument("--end_idx", type=int, default=150, help="结束行号")
    parser.add_argument("--tag", type=str, default="causalrca-codex", help="输出标签")
    parser.add_argument("--dataset_root", type=str, default=None, help="数据集根目录")
    parser.add_argument("--output_root", type=str, default=None, help="输出根目录")
    parser.add_argument("--no_diagnostics", action="store_true", help="不保存逐行诊断JSON")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    setup_logging(args.log_level)

    config = AgentLoopConfig(
        dataset_root=Path(args.dataset_root) if args.dataset_root else None,
        output_root=Path(args.output_root) if args.output_root else None,
    )

    df = run_dataset(
        dataset=args.dataset,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        tag=args.tag,
        config=config,
        save_diagnostics=not args.no_diagnostics,
    )

    # 打印汇总表格
    if not df.empty:
        print("\n" + "=" * 70)
        print("  评估结果汇总")
        print("=" * 70)
        for _, row in df.iterrows():
            status = "[OK]" if row.get("passed") and not row.get("failed") else "[X]"
            print(f"  {status} Row {row['row_id']:>3} | Task {row['task_index']} | Score: {row['score']}")


if __name__ == "__main__":
    main()
