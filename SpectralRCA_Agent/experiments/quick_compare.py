"""Quick visual demo: Traditional vs Spectral RCA full pipeline comparison."""
from __future__ import annotations

import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

from datetime import datetime, timedelta

from src.anomaly.metric_anomaly_expert import MetricAnomalyExpert
from src.config import SpectralRCAConfig
from src.data_loader import DataLoader
from src.graph.graph_refiner import SpectralGraphRefinementExpert
from src.ranking.root_cause_ranker import RootCauseRanker


def run_full_rca(expert, refiner, ranker, metric_dict, inc_start, inc_end, bl_start, gt_comp):
    evidence = expert.detect(metric_dict, inc_start, inc_end, bl_start, inc_start)
    edge_evidence = refiner.refine(evidence, metric_dict, incident_start=inc_start, incident_end=inc_end)
    ranked = ranker.rank(evidence, edge_evidence)

    gt_rank = None
    gt_score = 0.0
    for i, c in enumerate(ranked):
        if gt_comp and gt_comp in c.node_id:
            gt_rank = i + 1
            gt_score = c.root_score
            break

    detected = [e for e in evidence if e.final_anomaly_score > 0.5]
    gt_evidence = [e for e in evidence if gt_comp and gt_comp in e.node_id and e.final_anomaly_score > 0.3]
    best_gt = max(gt_evidence, key=lambda e: e.final_anomaly_score) if gt_evidence else None

    return {
        "detected_count": len(detected),
        "gt_rank": gt_rank,
        "gt_score": gt_score,
        "total_ranked": len(ranked),
        "best_gt_evidence": best_gt,
        "top5": [(c.node_id, c.root_score) for c in ranked[:5]],
        "kept_edges": sum(1 for e in edge_evidence if e.keep_edge),
        "total_edges": len(edge_evidence),
    }


def main():
    dataset_dir = r"d:\GitHubDownload\OpenRCA\dataset"
    dataset = "Bank"

    dl = DataLoader(dataset_dir)
    records = dl.load_records(dataset)
    folders = dl.list_date_folders(dataset)

    print("=" * 80)
    print("  SpectralRCA-Agent: 频域增强 vs 传统方法 完整RCA流水线对比")
    print("=" * 80)

    config_trad = SpectralRCAConfig()
    config_trad.enable_spectral_anomaly = False
    config_trad.enable_spectral_edge = False
    expert_trad = MetricAnomalyExpert(config_trad)
    refiner_trad = SpectralGraphRefinementExpert(config_trad)

    config_spec = SpectralRCAConfig()
    expert_spec = MetricAnomalyExpert(config_spec)
    refiner_spec = SpectralGraphRefinementExpert(config_spec)

    ranker = RootCauseRanker()

    all_trad_ranks = []
    all_spec_ranks = []
    total_cases = 0

    for date_folder in folders[:5]:
        date_part = date_folder.replace("_", "-")
        gt_comp = gt_reason = None
        inc_start = inc_end = None
        for _, row in records.iterrows():
            dt_str = str(row.get("datetime", ""))
            if date_part in dt_str:
                ts = float(row.get("timestamp", 0))
                dt = datetime.fromtimestamp(ts)
                inc_start = (dt - timedelta(minutes=15)).strftime("%Y-%m-%d %H:%M:%S")
                inc_end = (dt + timedelta(minutes=15)).strftime("%Y-%m-%d %H:%M:%S")
                gt_comp = row.get("component")
                gt_reason = row.get("reason")
                break

        if inc_start is None:
            continue

        metric_dict = dl.build_metric_series_dict(dataset, date_folder, resample_interval="60s")
        if not metric_dict:
            continue

        inc_dt_s = datetime.strptime(inc_start, "%Y-%m-%d %H:%M:%S")
        inc_dt_e = datetime.strptime(inc_end, "%Y-%m-%d %H:%M:%S")
        dur = inc_dt_e - inc_dt_s
        bl_start = (inc_dt_s - 3 * dur).strftime("%Y-%m-%d %H:%M:%S")

        trad_res = run_full_rca(expert_trad, refiner_trad, ranker, metric_dict, inc_start, inc_end, bl_start, gt_comp)
        spec_res = run_full_rca(expert_spec, refiner_spec, ranker, metric_dict, inc_start, inc_end, bl_start, gt_comp)

        total_cases += 1
        if trad_res["gt_rank"]:
            all_trad_ranks.append(trad_res["gt_rank"])
        if spec_res["gt_rank"]:
            all_spec_ranks.append(spec_res["gt_rank"])

        print(f"\n{'─' * 80}")
        print(f"  案例 {total_cases}: {date_folder}")
        print(f"  真实根因: {gt_comp} ({gt_reason})")
        print(f"{'─' * 80}")

        print(f"\n  [传统方法]")
        print(f"    检测异常数: {trad_res['detected_count']}")
        print(f"    候选边: {trad_res['total_edges']}, 保留: {trad_res['kept_edges']}")
        gt_rank_trad = trad_res['gt_rank'] if trad_res['gt_rank'] else '未命中'
        print(f"    GT排名: {gt_rank_trad} / {trad_res['total_ranked']}")
        if trad_res['gt_rank']:
            print(f"    GT得分: {trad_res['gt_score']:.4f}")
        print(f"    Top-5:")
        for i, (nid, sc) in enumerate(trad_res["top5"]):
            mark = " <-- GT" if (gt_comp and gt_comp in nid) else ""
            print(f"      {i+1}. {nid}: {sc:.4f}{mark}")

        print(f"\n  [频域增强方法]")
        print(f"    检测异常数: {spec_res['detected_count']}")
        print(f"    候选边: {spec_res['total_edges']}, 保留: {spec_res['kept_edges']}")
        gt_rank_spec = spec_res['gt_rank'] if spec_res['gt_rank'] else '未命中'
        print(f"    GT排名: {gt_rank_spec} / {spec_res['total_ranked']}")
        if spec_res['gt_rank']:
            print(f"    GT得分: {spec_res['gt_score']:.4f}")
        print(f"    Top-5:")
        for i, (nid, sc) in enumerate(spec_res["top5"]):
            mark = " <-- GT" if (gt_comp and gt_comp in nid) else ""
            print(f"      {i+1}. {nid}: {sc:.4f}{mark}")

        if spec_res["best_gt_evidence"]:
            bge = spec_res["best_gt_evidence"]
            print(f"\n  [GT组件频域分析]")
            print(f"    节点: {bge.node_id}")
            print(f"    传统分数: {bge.traditional_score:.4f}")
            print(f"    频域分数: {bge.spectral_score:.4f}")
            print(f"    融合分数: {bge.final_anomaly_score:.4f}")
            print(f"    频域类型: {bge.anomaly_type}")

        rank_improve = ""
        if trad_res['gt_rank'] and spec_res['gt_rank']:
            diff = trad_res['gt_rank'] - spec_res['gt_rank']
            if diff > 0:
                rank_improve = f"  >> 频域方法排名提升 {diff} 位! <<"
            elif diff < 0:
                rank_improve = f"  >> 传统方法排名更优 {-diff} 位 <<"
            else:
                rank_improve = f"  >> 排名相同 <<"
        print(f"\n{rank_improve}")

    print(f"\n{'=' * 80}")
    print(f"  汇总统计 ({total_cases} 个案例)")
    print(f"{'=' * 80}")

    if all_trad_ranks and all_spec_ranks:
        avg_trad = sum(all_trad_ranks) / len(all_trad_ranks)
        avg_spec = sum(all_spec_ranks) / len(all_spec_ranks)
        trad_top1 = sum(1 for r in all_trad_ranks if r <= 1) / len(all_trad_ranks)
        spec_top1 = sum(1 for r in all_spec_ranks if r <= 1) / len(all_spec_ranks)
        trad_top3 = sum(1 for r in all_trad_ranks if r <= 3) / len(all_trad_ranks)
        spec_top3 = sum(1 for r in all_spec_ranks if r <= 3) / len(all_spec_ranks)
        trad_top5 = sum(1 for r in all_trad_ranks if r <= 5) / len(all_trad_ranks)
        spec_top5 = sum(1 for r in all_spec_ranks if r <= 5) / len(all_spec_ranks)

        print(f"\n  {'指标':<20} {'传统方法':>12} {'频域增强':>12} {'提升':>12}")
        print(f"  {'─' * 56}")
        print(f"  {'平均GT排名':<20} {avg_trad:>12.2f} {avg_spec:>12.2f} {avg_trad - avg_spec:>+12.2f}")
        print(f"  {'Top-1 准确率':<20} {trad_top1:>12.1%} {spec_top1:>12.1%} {spec_top1 - trad_top1:>+12.1%}")
        print(f"  {'Top-3 准确率':<20} {trad_top3:>12.1%} {spec_top3:>12.1%} {spec_top3 - trad_top3:>+12.1%}")
        print(f"  {'Top-5 准确率':<20} {trad_top5:>12.1%} {spec_top5:>12.1%} {spec_top5 - trad_top5:>+12.1%}")

    print(f"\n  实验完成!")


if __name__ == "__main__":
    main()
