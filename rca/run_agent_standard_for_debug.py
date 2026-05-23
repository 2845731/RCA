import os
import sys
import json
import argparse
# 移除了 concurrent.futures 的导入

# NOTE: Previously this file forced HTTP_PROXY/HTTPS_PROXY to 127.0.0.1:7890.
# That breaks environments where the local proxy is not running. Proxy
# configuration should be controlled by the user's shell environment instead
# of being hard-coded here. If you need a proxy, set HTTP_PROXY/HTTPS_PROXY
# before launching the script.

# 将项目根目录加入 sys.path，这样后续可以用相对路径导入项目中的模块
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
sys.path.append("../.")
# 从 main.evaluate 导入 evaluate 函数用于评分
from main.evaluate import evaluate
# 从 rca.api_router 导入配置（例如 MODEL 等），在脚本末会打印所用模型名
from rca.api_router import configs

from datetime import datetime
from loguru import logger
from nbformat import v4 as nbf
import pandas as pd


def main(args, uid, dataset):
    """
    主运行函数：根据传入的数据集名称，读取指令和真实标签，创建/运行 RCA_Agent，收集预测、轨迹和 prompt

    参数：
    - args: argparse 解析后的命令行参数，包含范围、样本数、超时等控制参数
    - uid: 本次运行的唯一 id（通常为时间戳），用于在磁盘上保存文件
    - dataset: 数据集名字（例如 "Telecom"、"Bank" 或 "Market/cloudbed-1"）

    主要输出/副作用：
    - 在 test/monitor 下为每个样本保存 trajectory（ipynb）、prompt（json）、history（log）等
    - 在 test/result 下保存一个 csv，记录 instruction/prediction/groundtruth/score 等
    """
    # 在函数内部再导入 RCA_Agent 和 prompt 模块，按 dataset 选择不同的基础 prompt
    from rca.baseline.rca_agent.rca_agent import RCA_Agent
    import rca.baseline.rca_agent.prompt.agent_prompt as ap
    # 先给 bp 设一个默认值，避免静态分析提示未初始化引用
    bp = None
    if dataset == "Telecom":
        import rca.baseline.rca_agent.prompt.basic_prompt_Telecom as bp
    elif dataset == "Bank":
        import rca.baseline.rca_agent.prompt.basic_prompt_Bank as bp
    elif dataset == "Market/cloudbed-1" or dataset == "Market/cloudbed-2":
        import rca.baseline.rca_agent.prompt.basic_prompt_Market as bp
    # 如果没有匹配到已知的数据集，抛出明确错误（比未初始化引用更易理解）
    if bp is None:
        raise ValueError(f"Unsupported dataset: {dataset}")

    # 构造数据集和结果文件路径（基于 project_root 的绝对路径）
    # 支持 dataset 含子路径（例如 "Market/cloudbed-1"）
    dataset_parts = dataset.split("/")
    # 指向 dataset 下的 query.csv 和 record.csv（绝对路径）
    inst_file = os.path.join(project_root, "dataset", *dataset_parts, "query.csv")
    gt_file = os.path.join(project_root, "dataset", *dataset_parts, "record.csv")

    # eval_file: 存放本次模型预测结果的 csv，文件名里包含 tag 和模型名（绝对路径）
    model_name = configs['MODEL'].split('/')[-1]
    # 使用简单明了的 f-string 表示绝对路径（基于 project_root 和 dataset，避免复杂 join）
    eval_dir = f"{project_root}/test/result/{dataset}"
    eval_file = f"{eval_dir}/agent-{args.tag}-{model_name}.csv"

    # 观测（monitor）保存路径，用于保存每个样本的日志、轨迹和 prompt（绝对路径）
    obs_dir = f"{project_root}/test/monitor/{dataset}"
    obs_path = f"{obs_dir}/agent-{args.tag}-{model_name}"
    unique_obs_path = f"{obs_path}/{uid}"

    # 在读取 CSV 之前先确认文件存在，避免 Panda 抛出不友好的 FileNotFoundError
    if not os.path.exists(inst_file) or not os.path.exists(gt_file):
        raise FileNotFoundError(f"Please download the dataset first. Expected files: {inst_file}, {gt_file}")

    # 读取 instruction 和 groundtruth 的 csv 到 DataFrame
    instruct_data = pd.read_csv(inst_file)
    gt_data = pd.read_csv(gt_file)
    # （已在上面检查）

    # 创建保存轨迹、prompt、history 的目录（如果不存在）
    if not os.path.exists(f"{unique_obs_path}/history"):
        os.makedirs(f"{unique_obs_path}/history")
    if not os.path.exists(f"{unique_obs_path}/trajectory"):
        os.makedirs(f"{unique_obs_path}/trajectory")
    if not os.path.exists(f"{unique_obs_path}/prompt"):
        os.makedirs(f"{unique_obs_path}/prompt")
    # 如果评估结果文件不存在，则新建 DataFrame 并在写入到文件前创建父目录
    if not os.path.exists(eval_file):
        # 使用之前定义的绝对路径 eval_dir（基于 project_root）
        if not os.path.exists(eval_dir):
            os.makedirs(eval_dir)
        eval_df = pd.DataFrame(columns=["instruction", "prediction", "groundtruth", "passed", "failed", "score"])
    else:
        # 如果存在，读取已有结果以便追加
        eval_df = pd.read_csv(eval_file)

    # 初始化分数和计数统计结构
    scores = {
        "total": 0,
        "easy": 0,
        "middle": 0,
        "hard": 0,
    }
    nums = {
        "total": 0,
        "easy": 0,
        "middle": 0,
        "hard": 0,
    }

    # 打印所用数据集和模型（模型名取自 configs['MODEL'] 的最后一部分）
    logger.info(f"Using dataset: {dataset}")
    logger.info(f"Using model: {configs['MODEL'].split('/')[-1]}")

    # 遍历 instruction 列表（DataFrame.iterrows 返回 (idx, row)）
    for idx, row in instruct_data.iterrows():

        # 支持只跑子区间：如果索引小于 start_idx 就跳过；如果大于 end_idx 则结束循环
        if idx < args.start_idx:
            continue
        if idx > args.end_idx:
            break

        # 从行中取出需要的字段
        instruction = row["instruction"]
        task_index = row["task_index"]
        scoring_points = row["scoring_points"]
        # task_index 形如 'task_1'，这里取下划线后面的数字作为 id
        task_id = int(task_index.split('_')[1])
        best_score = 0

        # 根据 task_id 把任务分为 easy/middle/hard 三类（用于统计）
        catalog = None
        if task_id <= 3:
            catalog = "easy"
        elif task_id <= 6:
            catalog = "middle"
        elif task_id <= 7:
            catalog = "hard"
        # 如果 task_id 超出预期范围，抛出错误以避免后续使用未定义的 catalog
        if catalog is None:
            raise ValueError(f"Unexpected task_id: {task_id}")

        # 对每条 instruction 做 args.sample_num 次采样（可以用于多次随机 seed 运行）
        # 在循环之前先初始化临时统计，防止静态分析提示未定义
        temp_scores = scores.copy()
        temp_nums = nums.copy()
        for i in range(args.sample_num):
            # 为每次采样构造唯一 id（包含 uid、样本 idx 和采样序号 i）
            uuid = uid + f"_#{idx}-{i}"
            nb = nbf.new_notebook()  # 使用 nbformat 创建 notebook 对象来保存轨迹
            nbfile = f"{unique_obs_path}/trajectory/{uuid}.ipynb"
            promptfile = f"{unique_obs_path}/prompt/{uuid}.json"
            logfile = f"{unique_obs_path}/history/{uuid}.log"

            # 重新配置 logger 输出：先移除已有 handler，然后添加 stdout 和文件两个 handler
            logger.remove()
            logger.add(sys.stdout, colorize=True, enqueue=True, level="INFO")
            logger.add(logfile, colorize=True, enqueue=True, level="INFO")
            logger.debug('\n' + "#" * 80 + f"\n{uuid}: {task_index}\n" + "#" * 80)

            try:
                # 创建 agent（传入 agent_prompt 和 basic_prompt）
                agent = RCA_Agent(ap, bp)

                # 调试版本：直接调用 agent.run()，不使用线程池
                # 这样可以更方便地调试 agent 内部的执行过程
                prediction, trajectory, prompt = agent.run(
                    instruction,
                    logger,
                    max_step=args.controller_max_step,
                    max_turn=args.controller_max_turn
                )

                # 将 trajectory（每个 step 含 code 和 result）写入 notebook（code cell + markdown cell）
                for step in trajectory:
                    code_cell = nbf.new_code_cell(step['code'])
                    result_cell = nbf.new_markdown_cell(f"```\n{step['result']}\n```")
                    nb.cells.append(code_cell)
                    nb.cells.append(result_cell)
                # 把 notebook 写入磁盘为 ipynb 文件
                with open(nbfile, 'w', encoding='utf-8') as f:
                    json.dump(nb, f, ensure_ascii=False, indent=4)
                logger.info(f"Trajectory has been saved to {nbfile}")

                # 保存 agent 使用的 prompt（messages 列表）到 json 文件
                with open(promptfile, 'w', encoding='utf-8') as f:
                    json.dump({"messages": prompt}, f, ensure_ascii=False, indent=4)
                logger.info(f"Prompt has been saved to {promptfile}")

                # 将当前预测追加到 eval_df 中（先写入基本信息，score 等为 N/A，后面再评估并更新）
                new_eval_df = pd.DataFrame([{"row_id": idx,
                                             "task_index": task_index,
                                             "instruction": instruction,
                                             "prediction": prediction,
                                             # groundtruth 将 groundtruth DataFrame 中除 description 外的列拼接成多行字符串
                                             "groundtruth": '\n'.join(
                                                 [f'{col}: {gt_data.iloc[idx][col]}' for col in gt_data.columns if
                                                  col != 'description']),
                                             "passed": "N/A",
                                             "failed": "N/A",
                                             "score": "N/A"}])
                eval_df = pd.concat([eval_df, new_eval_df],
                                    ignore_index=True)
                # 立即把追加结果写回 csv（防止中途异常导致数据丢失）
                eval_df.to_csv(eval_file,
                               index=False)

                # 使用 evaluate 函数根据预测和 scoring_points 计算通过/失败的 criteria 以及得分
                passed_criteria, failed_criteria, score = evaluate(prediction, scoring_points)

                # 打印日志，更新 best_score
                logger.info(f"Prediction: {prediction}")
                logger.info(f"Scoring Points: {scoring_points}")
                logger.info(f"Passed Criteria: {passed_criteria}")
                logger.info(f"Failed Criteria: {failed_criteria}")
                logger.info(f"Score: {score}")
                best_score = max(best_score, score)

                # 将评估结果回写到 eval_df 最后一行（刚刚追加的那一行）
                eval_df.loc[eval_df.index[-1], "passed"] = '\n'.join(passed_criteria)
                eval_df.loc[eval_df.index[-1], "failed"] = '\n'.join(failed_criteria)
                eval_df.loc[eval_df.index[-1], "score"] = score
                eval_df.to_csv(eval_file,
                               index=False)

                # 更新临时统计（先 copy 然后修改，之后在循环外赋回 scores/nums）
                temp_scores[catalog] += best_score
                temp_scores["total"] += best_score
                temp_nums[catalog] += 1
                temp_nums["total"] += 1

            # 移除了 TimeoutError 的异常处理，因为直接调用没有超时机制
            except Exception as e:
                # 捕获其他异常，方便调试
                logger.error(f"Error occurred: {e}")
                import traceback
                traceback.print_exc()  # 打印详细的异常堆栈信息，方便调试
                continue

        # 在完成对当前 instruction 的所有采样后，把临时统计复制回主统计
        scores = temp_scores
        nums = temp_nums


if __name__ == "__main__":

    # 通过时间戳构造 uid（用于区分不同运行）
    uid = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Bank")
    parser.add_argument("--sample_num", type=int, default=1)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=150)
    parser.add_argument("--controller_max_step", type=int, default=25)
    parser.add_argument("--controller_max_turn", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=600)  # 保留这个参数但不使用
    parser.add_argument("--tag", type=str, default='rca-debug')  # 修改默认 tag 以便区分调试版本
    parser.add_argument("--auto", type=bool, default=False)

    args = parser.parse_args()

    # 如果开启 auto 模式，会对一组预定义的数据集依次运行 main
    if args.auto:
        print(f"Auto mode is on. Model is fixed to {configs['MODEL']}")
        datasets = ["Bank", "Market/cloudbed-1", "Market/cloudbed-2", "Telecom"]
        for dataset in datasets:
            main(args, uid, dataset)
    else:
        dataset = args.dataset
        main(args, uid, dataset)
