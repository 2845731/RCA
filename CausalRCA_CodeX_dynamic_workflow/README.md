# CausalRCA_CodeX

`CausalRCA_CodeX` 是一个面向微服务/数据库/缓存/中间件场景的多智能体根因定位项目。项目输入用户查询指令和一段故障窗口内的 telemetry 数据，经过异常检测、关联分析、候选筛选、因果图构建、干预排序和反事实仲裁，最终输出：

- `root cause occurrence datetime`
- `root cause component`
- 可选的根因原因说明 `reason`

项目核心目标不是为某个测试集写死规则，而是把根因定位拆成多个可解释、可诊断、可迭代优化的因果推理 Agent。每个 Agent 只保留少量真正重要的分数，并通过统一的工作区 `workspace` 传递中间结果，避免早期版本中大量重复分数相互污染。

## 1. 项目定位

故障根因定位通常面临三类困难：

1. **异常不等于根因**：下游服务、数据库、缓存都可能表现异常，但真正的源头通常只有少数几个组件。
2. **日志、指标、调用链信息不完整**：单一数据源容易缺边、缺指标或时间偏移。
3. **因果方向容易混淆**：例如 `Tomcat -> Mysql` 的调用链中，Mysql 异常可能影响 Tomcat，但 Tomcat 流量异常也可能传导到 Mysql。

因此本项目采用 multi-agent 流程：

```text
query.csv / telemetry
        |
        v
DataAgent
        |
        v
AssociationAgent
        |
        v
FaultIdentificationAgent
        |
        v
CausalGraphAgent + LogTopologyAgent
        |
        v
InterventionAgent
        |
        v
CounterfactualAgent
        |
        v
Final prediction / EvaluationAgent
```

每个 Agent 对应一个明确的建模问题：

| Agent | 解决的问题 | 核心输出 |
| --- | --- | --- |
| `DataAgent` | 读取数据、切分时间窗口、建立全日阈值 | `data_layer` |
| `AssociationAgent` | 判断哪些组件在故障窗口内异常 | `anomalies`, `association_scores` |
| `FaultIdentificationAgent` | 从异常组件中筛出根因候选 | `candidate_components`, `candidate_scores` |
| `CausalGraphAgent` | 构建组件间因果传播图 | `causal_graph`, `edge_scores` |
| `LogTopologyAgent` | 从日志补全调用拓扑 | `log_topology_layer` |
| `InterventionAgent` | 估计“干预某候选后能解释多少异常” | `RootCauseScore`, `ExplainScore` |
| `CounterfactualAgent` | 做最终反事实仲裁和时间定位 | `FinalScore`, prediction |
| `EvaluationAgent` | 仅在有标签时诊断错误原因 | `evaluation_layer` |

## 2. 目录结构

```text
CausalRCA_CodeX/
├── README.md
├── run_test.py                       # 调试入口，逐条测试并打印详细日志
├── run_causalrca_codex.py            # 标准 CLI 入口
├── causalrca_codex/
│   ├── config.py                     # 全局配置
│   ├── dataset.py                    # 数据集读取、query 构造、failure_count 推断
│   ├── runner.py                     # 批量运行入口
│   ├── orchestrator.py               # 多 Agent 编排器
│   ├── workspace.py                  # 跨 Agent 状态容器
│   ├── agents/
│   │   ├── data_agent.py
│   │   ├── association_agent.py
│   │   ├── fault_identification_agent.py
│   │   ├── causal_graph_agent.py
│   │   ├── log_topology_agent.py
│   │   ├── intervention_agent.py
│   │   ├── counterfactual_agent.py
│   │   └── evaluation_agent.py
│   ├── core/
│   │   ├── evidence.py               # 本地证据、原因证据、组件画像
│   │   ├── graph_ops.py              # 因果图路径和图操作
│   │   ├── metrics.py                # 指标异常检测工具
│   │   ├── reasoning.py              # 语义原因、类型先验
│   │   └── telemetry.py              # telemetry 文件加载
│   └── skills/
│       └── log_topology_agent_skill.md
└── output/
    └── <timestamp>/
        ├── run_logs/run.log
        └── causal_graphs/
```

## 3. 快速运行

### 3.1 准备数据

默认数据集结构类似：

```text
/root/workspace/ye/Root_Cause_Dataset/Bank/
├── query.csv
├── record.csv
└── telemetry/
    ├── metric/
    ├── trace/
    └── log/
```

`query.csv` 提供用户查询指令，例如“某时间段内某业务出现异常，请定位根因”。`record.csv` 通常包含 scoring points，用于评估预测时间和组件是否匹配。

### 3.2 调试运行

`run_test.py` 更适合开发调试，因为它会打印每条样本的完整过程和最终评分。

```powershell
cd D:\GitHubDownload\OpenRCA\CausalRCA_CodeX
python run_test.py --dataset Bank --start_idx 0 --end_idx 1
```

注意：`run_test.py` 的 `--end_idx` 是右开区间。例如 `--start_idx 0 --end_idx 1` 只运行第 0 条。

每次运行都会生成新的日志目录：

```text
CausalRCA_CodeX/output/<YYYY-MM-DD_HH-MM-SS>/run_logs/run.log
```

### 3.3 标准批量运行

`run_causalrca_codex.py` 会调用 `runner.py`，适合批量生成结果文件。

```powershell
cd D:\GitHubDownload\OpenRCA\CausalRCA_CodeX
python run_causalrca_codex.py --dataset Bank --start_idx 0 --end_idx 10
```

注意：标准 runner 中的 `--end_idx` 是闭区间，`--start_idx 0 --end_idx 10` 会运行 0 到 10 共 11 条。

批量运行会额外生成：

```text
CausalRCA_CodeX/output/<timestamp>/dataset/<tag>.csv
CausalRCA_CodeX/output/<timestamp>/dataset/<tag>-diagnostics/row_XXXX.json
```

## 4. 输入与输出契约

### 4.1 Query 构造

`dataset.py` 会从 `query.csv` 和 `record.csv` 构造统一 query：

```python
{
    "instruction": "...用户原始查询...",
    "dataset": "Bank",
    "row_index": 0,
    "telemetry_path": ".../telemetry",
    "target_fields": ["root cause occurrence datetime", "root cause component"],
    "candidate_components": [...],
    "candidate_reasons": [...],
    "failure_count": 1
}
```

`failure_count` 会从查询文本中推断，支持数字和英文单词，例如 `two failures`、`three failures`、`multiple root causes`。

### 4.2 最终输出

最终预测会格式化为 JSON 风格字段：

```json
{
  "root cause occurrence datetime": "2021-03-09 15:25:00",
  "root cause component": "Tomcat01"
}
```

如果是多根因，`root cause component` 和时间字段可以是列表；否则是单个字符串。

## 5. 全局工作区 Workspace

`workspace.py` 是所有 Agent 共享的状态容器。每个 Agent 只读取前序层需要的信息，并写入自己的 layer。

主要 layer 如下：

| Layer | 来源 | 内容 |
| --- | --- | --- |
| `data_layer` | `DataAgent` | 指标序列、trace/log 窗口、全日阈值、质量分 |
| `association_layer` | `AssociationAgent` | 组件异常分数、异常段、原因证据 |
| `fault_id_layer` | `FaultIdentificationAgent` | 根因候选、候选证据分 |
| `causal_graph_layer` | `CausalGraphAgent` | 因果图、边权、节点类型 |
| `intervention_layer` | `InterventionAgent` | Top-K 候选和 RCS/ES |
| `counterfactual_layer` | `CounterfactualAgent` | CFS、ReasonScore、FinalScore、最终预测 |
| `evaluation_layer` | `EvaluationAgent` | 错误类型诊断，只用于分析 |

当前版本已经精简跨 Agent 传递分数。原则是：每一步只传递下一步真正需要的少数核心量，避免每个 Agent 都产生一堆相似分数。

## 6. 核心分数契约

当前代码保留的核心分数如下。

| 阶段 | 分数 | 含义 |
| --- | --- | --- |
| Association | `AnomalyScore` | 组件在故障窗口内的异常强度 |
| Association | `local_root_evidence` | 组件自身指标是否具有根因型本地证据 |
| Association | `reason_evidence` | 候选原因与组件指标/日志证据的匹配程度 |
| FaultIdentification | `CandidateEvidence` | 候选根因本地可信度 |
| CausalGraph | `EdgeWeight` | 组件间因果传播边权 |
| Intervention | `ExplainScore` | 候选通过因果路径解释其他异常的能力 |
| Intervention | `RootCauseScore` | 干预层综合根因分 |
| Counterfactual | `CounterfactualScore` | 反事实解释分，考虑竞争父节点折扣 |
| Counterfactual | `ReasonScore` | 候选原因证据分 |
| Counterfactual | `FinalScore` | 最终排序分 |

### 6.1 候选证据分

`FaultIdentificationAgent` 使用：

```text
CandidateEvidence = 0.65 * AnomalyScore + 0.35 * LocalEvidence
```

其中：

- `AnomalyScore` 来自 `AssociationAgent`
- `LocalEvidence` 来自 `core/evidence.py` 的 `local_root_evidence`

这个分数强调：根因组件不仅要异常，还要有自己的本地根因证据。这样可以降低“下游被影响组件异常更明显”导致的误判。

### 6.2 干预层解释分

`InterventionAgent` 对每个候选组件 `c` 计算：

```text
RootCauseScore(c)
  = 0.40 * Evidence(c)
  + 0.25 * Source(c)
  + 0.20 * Time(c)
  + 0.15 * ExplainScore(c)
```

含义：

- `Evidence(c)`：候选自身证据，即 `CandidateEvidence`
- `Source(c)`：候选是否像异常传播源，而不是单纯下游受害者
- `Time(c)`：候选异常是否更早出现
- `ExplainScore(c)`：候选通过因果图解释其他异常节点的能力

`ExplainScore` 使用图路径上的 bottleneck 思想。如果 `c` 到某异常节点 `v` 有多条路径，取最大路径强度：

```text
PathStrength(path) = min(edge_weight_1, edge_weight_2, ..., edge_weight_k)
MaxPathStrength(c, v) = max PathStrength(path)
ExplainScore(c) = weighted_average_v MaxPathStrength(c, v)
```

这样可以避免一条路径中某条弱边被其他强边掩盖。

### 6.3 反事实最终分

`CounterfactualAgent` 做最终仲裁：

```text
FinalScore(c)
  = 0.55 * RootCauseScore(c)
  + 0.25 * CounterfactualScore(c)
  + 0.20 * ReasonScore(c)
```

其中 `CounterfactualScore` 是经过竞争父节点折扣后的反事实解释分：

```text
CounterfactualScore(c) = ContextualExplainScore(c) * (1 - 0.70 * IncomingRatio(c))
```

直观解释：

- 如果候选能解释大量其他异常，`ContextualExplainScore` 高。
- 如果候选本身又被多个更强父节点解释，`IncomingRatio` 高，说明它更可能是中间传播节点或下游节点，需要折扣。
- `ReasonScore` 用于把用户查询中的故障原因和候选组件的本地证据对齐。

当前跨 Agent Top-K 每行保持为精简结构：

```python
{
    "component": "Redis01",
    "RootCauseScore": 0.72,
    "ExplainScore": 0.65,
    "CounterfactualScore": 0.58,
    "ReasonScore": 0.40,
    "FinalScore": 0.64
}
```

## 7. Agent 详细流程

### 7.1 DataAgent

`DataAgent` 负责把原始 telemetry 转换成后续可用的结构化数据。

主要步骤：

1. 从 query 中解析故障时间窗口。
2. 调用 `load_day_frames` 读取同一天的 metric、trace、log。
3. 用全日数据计算全局阈值，避免只看故障窗口导致阈值漂移。
4. 在故障窗口内抽取组件 KPI 序列。
5. 根据 `time_window_extension_minutes` 适度扩展窗口，容纳故障前兆。
6. 写入 trace/log 窗口数据、全日 trace、组件延迟统计和数据质量分。

数据质量分用于 Recovery 机制。如果某一阶段因为缺数据导致质量过低，Orchestrator 会根据 `recovery_budget` 触发补救或降级路径。

### 7.2 AssociationAgent

`AssociationAgent` 判断哪些组件出现异常，并建立本地证据。

常见指标方向：

- CPU、内存、磁盘 I/O、延迟、错误率：通常高值异常
- 可用连接数、成功率、吞吐下降类指标：可能低值异常

异常检测使用全日阈值和故障窗口序列。对每个组件，Agent 会得到：

```text
raw_high = high-side deviation
raw_low  = low-side deviation
AnomalyScore = normalized deviation strength
```

然后 `core/evidence.py` 会把指标名、组件类型、用户原因语义映射到更抽象的证据族，例如：

- `cpu`
- `memory`
- `disk_io`
- `network`
- `latency`
- `connection`
- `error`

当前版本只向后传递两个压缩后的证据：

```text
local_root_evidence
reason_evidence
```

早期版本中的 `family_scores`、`dominant_family`、`specificity`、`diversity` 等中间分数不再跨 Agent 传递，只在本地内部计算时使用。

### 7.3 FaultIdentificationAgent

`FaultIdentificationAgent` 的任务是从异常组件中挑选根因候选，而不是直接给最终答案。

它会读取：

- `association_scores`
- `local_root_evidence`
- 组件类型
- 查询中推断的 `failure_count`

核心公式：

```text
CandidateEvidence = 0.65 * AnomalyScore + 0.35 * LocalEvidence
```

候选选择逻辑遵循通用原则：

1. 先按组件层级聚合候选，例如应用、数据库、缓存、中间件。
2. 每层的 layer score 取该层组件的最大 `CandidateEvidence`。
3. 选择主层，同时保留证据接近的竞争层。
4. 单根因任务优先保持候选集紧凑，多根因任务会扩大候选宽度。
5. 如果单个候选明显强于其他候选，可触发 single-root shortcut。

这个阶段不使用标签，也不根据测试集错误样本写修正规则。

### 7.4 LogTopologyAgent

`LogTopologyAgent` 是为了补齐 trace 缺失边而加入的拓扑智能体。它从日志中挖掘服务和数据库/缓存之间的调用关系。

主要策略：

| 策略 | 日志线索 | 用途 |
| --- | --- | --- |
| A | DBCP2 datasource startup | 识别 Java 服务连接的 MySQL 数据源 |
| B | Jedis Redis pool | 识别 Redis 连接 |
| C | JDBC URL | 从 URL 中解析数据库主机 |
| D | connection error | 弱信号，辅助确认依赖 |
| E | access log caller | 从访问日志估计调用方 |

输出写入：

```python
workspace.log_topology_layer = {
    "call_counts": {...},
    "raw_edges": [...],
    "matched_edges": [...],
    "unmatched_edges": [...],
    "datasource_mapping": {...},
    "extraction_stats": {...},
    "status": "MERGED"
}
```

随后 `CausalGraphAgent` 会把 log-derived edges 合并进主调用图。日志边不会被当成绝对事实，而是作为一种中等可靠度的拓扑证据。

### 7.5 CausalGraphAgent

`CausalGraphAgent` 构建故障传播因果图。

主要步骤：

1. 从 trace 中统计调用次数 `call_counts`。
2. 调用 `infer_missing_dependencies` 补全缺失依赖。
3. 合并 `LogTopologyAgent` 发现的日志拓扑边。
4. 选择与候选和异常相关的节点。
5. 对每条边计算 `EdgeWeight`。
6. 导出机器可读和人类可读的因果图文件。

边方向采用故障传播方向：

```text
caller -> callee 表示调用依赖
callee 异常可以沿依赖影响 caller
```

因此在因果传播图中要特别注意方向含义：数据库故障常常会影响调用它的服务，而不是只看调用链原始方向。

内部边权融合多个证据：

```text
s_time  = temporal support
s_corr  = correlation support
s_prior = component-type prior

TelemetrySupport = 0.35 * s_time + 0.30 * s_corr + 0.35 * s_prior
```

其中：

- `s_time`：父节点异常是否早于子节点异常
- `s_corr`：两个组件 KPI 序列在小 lag 范围内的最大相关性
- `s_prior`：组件类型间传播先验，例如 DB/Cache 到 Service 的影响更常见

调用链或日志拓扑提供可靠度：

```text
TraceReliability = 1 - exp(-call_count / lambda_call_count)
```

最终边权是拓扑可靠度和 telemetry support 的融合。输出中只保留精简字段：

```python
{
    "weight": 0.73,
    "call_count": 12,
    "observed_trace": True,
    "from_log_topology": False,
    "confirmed_by_log": True
}
```

日志中会显示：

```text
EdgeWeight = fused topology + telemetry support
source = trace / log / inferred / trace+log
```

### 7.6 InterventionAgent

`InterventionAgent` 模拟“如果把某个候选组件作为根因干预掉，它能解释多少异常”。

输入：

- `candidate_components`
- `candidate_scores`
- `causal_graph`
- `edge_scores`
- `association_scores`

输出 Top-K：

```python
{
    "component": "Tomcat01",
    "RootCauseScore": 0.81,
    "ExplainScore": 0.66
}
```

核心公式：

```text
RootCauseScore
  = 0.40 * Evidence
  + 0.25 * Source
  + 0.20 * Time
  + 0.15 * ExplainScore
```

其中：

- `Evidence` 是候选自己的 `CandidateEvidence`
- `Source` 反映候选是否是异常传播源
- `Time` 反映候选是否早于其他组件异常
- `ExplainScore` 反映候选通过图路径解释其他异常的能力

这个阶段不会直接输出最终结果，只负责给 `CounterfactualAgent` 提供高质量 Top-K。

### 7.7 CounterfactualAgent

`CounterfactualAgent` 是最终决策层。它会回答一个反事实问题：

> 如果这个候选不是根因，当前异常模式还能被其他候选更好解释吗？

它会读取：

- Intervention Top-K
- causal graph
- association anomalies
- reason evidence
- failure_count

最终排序公式：

```text
FinalScore
  = 0.55 * RootCauseScore
  + 0.25 * CounterfactualScore
  + 0.20 * ReasonScore
```

`CounterfactualScore` 会对被强父节点解释的候选进行折扣：

```text
CounterfactualScore = ContextualExplainScore * (1 - 0.70 * IncomingRatio)
```

`ReasonScore` 来自候选组件的 `reason_evidence`，用于判断候选是否符合用户查询中的故障原因。例如查询提到 `high disk I/O read usage`，则 MySQL、磁盘读写相关指标会获得更高原因证据，但仍需和传播图、时间、候选证据一起综合判断。

#### 时间定位

最终时间不是简单取 KPI 峰值，而是优先定位异常 onset。

对于候选组件的异常段：

1. 如果异常开始点已经接近峰值，取段起点。
2. 否则寻找第一个显著上升点。
3. 如果没有明显上升点，再退回峰值点。
4. 如果候选无有效异常段，退回查询窗口或全局异常时间。

这样可以降低把“故障结果峰值”误当成“根因发生时间”的概率。

#### 多根因

如果 `failure_count > 1`，`CounterfactualAgent` 会选择多个 FinalScore 高且相互不完全冗余的候选。选择时会避免多个候选只是同一传播链上的重复下游节点。

### 7.8 EvaluationAgent

`EvaluationAgent` 只在传入 ground truth 时运行，用于开发阶段分析错误，不参与最终预测。

它会诊断常见错误类型：

| 错误类型 | 含义 |
| --- | --- |
| `AD-FN` | 异常检测漏掉真实根因 |
| `AD-FP-NOISE` | 噪声异常过多 |
| `FI-MISFILTER` | 候选筛选阶段过滤错 |
| `CG-MISSING-NODE` | 因果图缺失关键节点 |
| `INT-RANK` | 干预排序错误 |
| `CF-REASON` | 原因证据仲裁错误 |
| `CF-TIME` | 时间定位错误 |
| `TOOL-ERROR` | 工具或数据读取错误 |

这个 Agent 的定位是“解释模型为什么错”，而不是把标签反馈回模型调参。

## 8. 日志阅读方法

每轮运行都会在新目录中生成日志：

```text
CausalRCA_CodeX/output/<timestamp>/run_logs/run.log
```

建议按以下顺序阅读：

1. `Instruction`：用户原始查询，判断目标组件、时间、原因语义。
2. `DataAgent`：检查 telemetry 是否读到，时间窗口是否合理。
3. `AssociationAgent`：看真实根因组件是否被检测为异常。
4. `FaultIdentificationAgent`：看真实根因是否进入候选集。
5. `CausalGraphAgent`：看关键传播边是否存在，边方向是否合理。
6. `InterventionAgent`：看 Top-K 中 `RootCauseScore` 是否被下游组件压制。
7. `CounterfactualAgent`：看 `FinalScore`、`ReasonScore`、`CounterfactualScore` 的最终仲裁。
8. `FINAL RESULT`：查看模型输出、格式化 JSON、scoring points 和匹配结果。

最终结果块形如：

```text
======================================================================
FINAL RESULT
======================================================================
Original Instruction:
  ...

Model Output:
  Component : Tomcat01
  Time      : 2021-03-09 15:25:00
  Reason    : ...

Formatted Prediction (JSON):
  root cause occurrence datetime: 2021-03-09 15:25:00
  root cause component: Tomcat01

Ground Truth (Scoring Points):
  ...

Match Result:
  ...
```

因果图文件通常保存在：

```text
CausalRCA_CodeX/output/<timestamp>/causal_graphs/
```

调试时也可能在 `test/result/<dataset>/row_xxx_causal_graphs/` 下看到图文件。

## 9. 关键配置

配置集中在 `causalrca_codex/config.py`。

常用参数：

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `top_k` | `4` | Intervention 输出候选数量 |
| `threshold_percentile` | `95` | 高值异常阈值分位数 |
| `low_percentile` | `5` | 低值异常阈值分位数 |
| `severity_threshold` | `0.05` | 最小异常强度 |
| `min_fault_points` | `2` | 构成异常段所需最少点数 |
| `time_window_extension_minutes` | `5` | 故障窗口扩展分钟数 |
| `max_candidate_components` | `30` | 最大候选组件数量 |
| `max_path_depth` | `6` | 因果路径搜索深度 |
| `lambda_call_count` | `5` | trace 调用次数可靠度尺度 |
| `lambda_early` | `300` | 时间早发性衰减尺度 |
| `use_llm_reasoning` | `False` | 是否启用 LLM 推理 |
| `use_llm_edge_scoring` | `False` | 是否启用 LLM 边打分 |

当前默认是可复现的非 LLM 流程。若后续启用 LLM，应把 LLM 用作解释、检验和候选审阅，而不是让 LLM 直接替代结构化因果分数。

## 10. 算法设计原则

本项目优化遵循以下原则：

1. **机制优先**：改进异常检测、候选筛选、因果传播、反事实仲裁等通用机制。
2. **少量核心分数**：每个阶段只保留最有解释力的分数，降低重复特征叠加。
3. **多源证据融合**：指标、trace、日志、组件类型和用户原因语义共同参与判断。
4. **不使用测试标签反向设计规则**：EvaluationAgent 只用于错误诊断，不用于写死修正规则。
5. **因果方向明确**：区分异常强度、传播源、下游受害者和中间节点。
6. **可解释可诊断**：每个候选的分数来源都能在日志中追踪。

项目特别避免以下做法：

- 根据某个错误样本硬编码组件修正。
- 根据当前测试集搜索最优阈值。
- 通过堆叠大量 if-else 规则提高短期指标。
- 把偶然数据分布当成通用故障机制。

## 11. 初学者阅读代码路线

建议按这个顺序看源码：

1. `run_test.py`：理解一条样本如何被读取、运行和评分。
2. `orchestrator.py`：理解 Agent 执行顺序和 Recovery 机制。
3. `workspace.py`：理解跨 Agent 状态如何传递。
4. `agents/data_agent.py`：理解 telemetry 如何变成结构化输入。
5. `agents/association_agent.py` 和 `core/evidence.py`：理解异常和本地证据。
6. `agents/fault_identification_agent.py`：理解候选根因如何产生。
7. `agents/causal_graph_agent.py` 和 `agents/log_topology_agent.py`：理解因果图和日志拓扑补全。
8. `agents/intervention_agent.py`：理解根因候选如何排序。
9. `agents/counterfactual_agent.py`：理解最终根因和时间如何确定。
10. `agents/evaluation_agent.py`：理解错误诊断输出。

阅读时建议同时打开最新 `run.log`。日志中的每个阶段输出都能对应到相应 Agent 的代码，这样最容易建立“代码步骤 -> 中间结果 -> 最终预测”的完整理解。
