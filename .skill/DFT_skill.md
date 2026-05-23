你是一名资深 AIOps / RCA / Python 工程师。请为我实现一个“Spectral-RCA: 频域增强的异常检测与因果图精化框架”。该框架用于云原生系统 Root Cause Analysis，输入 metric 时间序列、故障时间窗口、正常基线窗口和候选调用图，输出异常 metric、精化后的故障传播因果图、根因排序和解释证据。

一、总体目标

实现两个核心模块：

1. MetricAnomalyExpert：
   在传统 metric 异常检测基础上加入频域异常检测。
   传统检测包括：超阈值、鲁棒 z-score、变化率、最大偏离。
   频域检测包括：DFT Total Energy、低频/中频/高频能量占比、主导频率、主导频率能量占比、频谱熵、incident 与 baseline 的频域偏离分数。
   输出每个 component.metric 的 anomaly_score、traditional_score、spectral_score、anomaly_type 和 evidence。

2. SpectralGraphRefinementExpert：
   在候选故障传播图上对每条有向候选边 u -> v 做频域传播一致性验证。
   使用：时域滞后相关、频谱形状相似度、主导频率一致性、频域相位滞后一致性、图结构一致性。
   输出 refined_graph，每条边包含 final_edge_weight 和详细 evidence。
   方向性由候选传播方向、时域滞后、频域相位滞后决定。
   图拉普拉斯或图结构一致性只用于无向化后的结构平滑检查，不用于直接学习方向。

二、输入数据格式

metric_df:
- timestamp: 时间戳
- component: 组件名，例如 ProductDB、OrderService、CheckoutService
- metric_name: 指标名，例如 db_latency、order_latency、cpu_usage
- value: 数值

candidate_edges:
列表，每条边包含：
- source: 源节点，格式为 component.metric_name
- target: 目标节点，格式为 component.metric_name
- prior_weight: 先验权重，默认为 1.0
- relation_type: invoke / on / semantic / historical

配置参数：
- incident_start_time
- incident_end_time
- baseline_start_time
- baseline_end_time
- resample_interval，默认 60s
- max_lag，默认 min(5, incident_length // 4)
- pruning_threshold，默认 0.30
- eps，默认 1e-8

三、代码结构

请实现如下文件结构：

src/
  data_loader.py
  preprocessing.py
  anomaly/
    traditional.py
    spectral.py
    metric_anomaly_expert.py
  graph/
    spectral_edge.py
    graph_consistency.py
    graph_refiner.py
  ranking/
    root_cause_ranker.py
  pipeline.py
  schemas.py
  utils.py
experiments/
  run_pipeline.py
  run_ablation.py
tests/
  test_spectral_features.py
  test_anomaly_expert.py
  test_graph_refiner.py

四、数据结构

在 schemas.py 中定义 dataclass：

MetricAnomalyEvidence:
- node_id: str
- traditional_score: float
- spectral_score: float
- final_anomaly_score: float
- anomaly_type: str
- max_abs_robust_z: float
- max_change_rate: float
- max_deviation_score: float
- total_energy: float
- spectral_energy_z: float
- low_ratio: float
- mid_ratio: float
- high_ratio: float
- dominant_freq_index: int
- dominant_freq_energy_ratio: float
- spectral_entropy: float
- quality_flag: str
- explanation: str

EdgeEvidence:
- source: str
- target: str
- prior_weight: float
- node_anomaly_factor: float
- best_lag: int
- lag_corr: float
- spectral_shape_similarity: float
- dominant_freq_match: float
- phase_lag_consistency: float
- graph_consistency: float
- final_edge_weight: float
- keep_edge: bool
- explanation: str

RootCauseCandidate:
- node_id: str
- root_score: float
- anomaly_score: float
- out_evidence: float
- in_evidence: float
- onset_time: optional timestamp
- explanation: str

五、预处理要求

1. 对每个 component.metric 单独处理。
2. 将 timestamp 转成 pandas datetime。
3. 按 resample_interval 重采样，默认使用 mean。
4. 缺失值处理：
   - 缺失比例 > 30%：quality_flag = "low_quality"，但不要直接丢弃；
   - 中间缺失：linear interpolation；
   - 开头/结尾缺失：forward fill / backward fill。
5. 分别切出 baseline_series 和 incident_series。
6. 如果 incident_series 长度 < 8，则不做频域检测，spectral_score = 0，quality_flag 标记为 "too_short_for_spectral"。

六、传统异常检测实现

在 anomaly/traditional.py 中实现：

1. robust_median_mad(series)
   返回 median 和 MAD。

2. robust_z_scores(incident, baseline)
   使用公式：
   robust_z = (x - median(baseline)) / (1.4826 * MAD(baseline) + eps)

3. threshold_score(incident, baseline, optional_threshold=None)
   如果 optional_threshold 存在，判断 max(incident) 是否超过该阈值。
   如果没有人工阈值，使用 baseline 的 0.99 quantile 作为动态阈值。

4. change_rate_score(incident, baseline)
   计算相邻点变化率：
   abs(x_t - x_{t-1}) / (abs(x_{t-1}) + eps)
   将 incident 的最大变化率与 baseline 的变化率分布比较，输出 0 到 1 分数。

5. max_deviation_score(incident, baseline)
   计算 incident 中最大偏离 baseline median 的程度，并归一化。

6. traditional_anomaly_score(...)
   融合：
   threshold_score: 0.20
   z_score: 0.35
   change_rate_score: 0.25
   max_deviation_score: 0.20

七、频域异常检测实现

在 anomaly/spectral.py 中实现：

1. center_signal(x)
   返回 x - mean(x)。

2. compute_fft_power(x)
   - 输入一维 incident 序列；
   - 先减均值；
   - 使用 np.fft.rfft；
   - power = abs(X) ** 2；
   - 去掉 k=0 常量成分；
   - 返回 X, power_no_dc, freqs。

3. compute_spectral_features(x)
   返回：
   - total_energy = sum(power_no_dc)
   - low_ratio
   - mid_ratio
   - high_ratio
   - dominant_freq_index
   - dominant_freq_energy_ratio
   - spectral_entropy

其中低频/中频/高频按 power_no_dc 长度三等分。若长度太短，则尽量合理切分，至少返回有效值。

4. split_baseline_windows(baseline, window_len)
   将 baseline 切成多个与 incident 等长的窗口。
   如果 baseline 不足以切多个窗口，则至少使用一个 baseline 窗口。

5. compute_baseline_spectral_stats(baseline_windows)
   对每个 baseline window 计算 spectral features。
   返回每个特征的 median 和 MAD。

6. spectral_anomaly_score(incident, baseline)
   - 计算 incident spectral features；
   - 计算 baseline window spectral features；
   - 对 total_energy 使用 log 后做 robust z；
   - 对 low_ratio、mid_ratio、high_ratio、dominant_freq_energy_ratio、spectral_entropy 分别和 baseline 做 robust z；
   - 输出 spectral_score，范围 0 到 1。

建议：
   energy_score = sigmoid((spectral_energy_z - 3.0) / 1.0)
   low_shift_score = sigmoid((abs(low_ratio_z) - 2.5) / 1.0)
   high_shift_score = sigmoid((abs(high_ratio_z) - 2.5) / 1.0)
   periodic_score = sigmoid((dominant_freq_energy_ratio_z - 2.5) / 1.0)
   entropy_shift_score = sigmoid((abs(entropy_z) - 2.5) / 1.0)

融合：
   spectral_score = 0.40 * energy_score
                  + 0.20 * low_shift_score
                  + 0.20 * high_shift_score
                  + 0.10 * periodic_score
                  + 0.10 * entropy_shift_score

7. classify_spectral_anomaly(features, z_values)
   规则：
   - 如果 spectral_energy_z >= 3.0 且 low_ratio >= 0.55，anomaly_type = "slow_trend_anomaly"
   - 如果 spectral_energy_z >= 3.0 且 high_ratio >= 0.45，anomaly_type = "fast_burst_or_jitter_anomaly"
   - 如果 spectral_energy_z >= 3.0 且 dominant_freq_energy_ratio >= 0.60，anomaly_type = "periodic_oscillation_anomaly"
   - 如果 spectral_energy_z >= 3.0，anomaly_type = "mixed_spectral_anomaly"
   - 否则 anomaly_type = "no_strong_spectral_anomaly"

八、MetricAnomalyExpert 实现

在 anomaly/metric_anomaly_expert.py 中实现类 MetricAnomalyExpert。

主要方法：

detect(metric_df, incident_start, incident_end, baseline_start, baseline_end)

流程：
1. 遍历每个 component.metric。
2. 预处理并切分 baseline / incident。
3. 调用 traditional.py 得到 traditional_score。
4. 调用 spectral.py 得到 spectral_score 和 spectral features。
5. final_anomaly_score = 0.6 * traditional_score + 0.4 * spectral_score。
6. 判断异常：
   - final_anomaly_score >= 0.85: high_confidence_anomaly
   - final_anomaly_score >= 0.70: candidate_anomaly
   - else: normal_or_weak
7. 输出 MetricAnomalyEvidence 列表。
8. explanation 必须用可读文本说明为什么异常，例如：
   "OrderService.order_latency shows high robust z-score and high-frequency spectral energy, suggesting burst/jitter anomaly."

九、边级频域传播验证实现

在 graph/spectral_edge.py 中实现：

1. lagged_correlation(xu, xv, max_lag)
   对候选边 u -> v，计算 u 领先 v 的相关性。
   对 lag 从 0 到 max_lag：
      corr = pearson_corr(xu[:-lag], xv[lag:])
   lag=0 时直接 corr(xu, xv)。
   返回 best_lag 和 best_lag_corr。
   负相关可以保留 absolute 版本和 signed 版本，但第一版使用正相关更稳。

2. spectral_shape_similarity(xu, xv)
   - 对 xu 和 xv 分别计算 power_no_dc；
   - 归一化成 pu 和 pv；
   - 计算余弦相似度；
   - 返回 0 到 1 的分数。

3. dominant_frequency_match(xu, xv, freq_tolerance=1)
   - 计算两个序列的 dominant_freq_index；
   - 如果差值 <= 1，则返回 1，否则返回 0；
   - 也可以返回连续分数：exp(-abs(dom_u-dom_v))。

4. phase_lag_consistency(xu, xv, max_allowed_lag)
   - 只有当 len(xu) >= 16 时启用；
   - 对 xu 和 xv 分别减均值，做 np.fft.rfft；
   - 计算两个序列共享高能量频率；
   - 对共享高能量频率计算 phase = angle(Xv * conj(Xu))；
   - lag = -phase / (2*pi*freq)；
   - 如果 0 <= lag <= max_allowed_lag，认为该频率支持 u -> v；
   - 返回支持比例。
   如果无法稳定计算，则返回 0.5，表示中性证据。

5. compute_edge_spectral_score(xu, xv)
   返回：
   edge_spectral_score =
       0.45 * spectral_shape_similarity
     + 0.20 * dominant_freq_match
     + 0.20 * phase_lag_consistency
     + 0.15 * normalized_lag_corr

十、图结构一致性实现

在 graph/graph_consistency.py 中实现：

1. build_spectral_profile(metric_evidence)
   对每个节点构造频域画像向量：
   [
     normalized_total_energy,
     low_ratio,
     mid_ratio,
     high_ratio,
     dominant_freq_energy_ratio,
     normalized_spectral_entropy
   ]

2. graph_consistency_score(profile_u, profile_v, eta=1.0)
   dist = l2_distance(profile_u, profile_v)
   score = exp(-eta * dist)
   返回 0 到 1。

注意：
- 不要直接用异常幅度差作为图一致性，因为根因节点与下游节点异常强度可能不同；
- 应优先使用频域形状向量。

十一、图精化实现

在 graph/graph_refiner.py 中实现类 SpectralGraphRefinementExpert。

输入：
- candidate_edges
- metric_series_dict
- anomaly_evidence_dict
- prior_weight
- pruning_threshold

流程：
1. 遍历每条候选边 u -> v。
2. 如果 u 或 v 缺少 metric 序列，标记为缺失证据，降低权重。
3. 取 incident window 中的 xu 和 xv。
4. 计算 node_anomaly_factor = sqrt(anomaly_score[u] * anomaly_score[v])。
5. 计算 lagged_correlation。
6. 计算 spectral_shape_similarity。
7. 计算 dominant_frequency_match。
8. 计算 phase_lag_consistency。
9. 计算 graph_consistency。
10. 计算 final_edge_weight：

final_edge_weight =
    prior_weight
  * node_anomaly_factor
  * normalized_lag_corr
  * edge_spectral_score
  * graph_consistency

11. 如果 final_edge_weight >= pruning_threshold，则 keep_edge = True，否则 False。
12. 输出 EdgeEvidence 列表和 refined_graph。

十二、根因排序实现

在 ranking/root_cause_ranker.py 中实现：

输入：
- refined_graph
- anomaly_evidence_dict

对每个节点 u：

out_evidence = sum(final_edge_weight[u,v] * anomaly_score[v] for v in successors(u))
in_evidence = sum(final_edge_weight[p,u] * anomaly_score[p] for p in predecessors(u))

base_root_score = anomaly_score[u] * (1 + out_evidence) / (1 + in_evidence)

如果有 onset_time，则加入 early factor：
root_score = base_root_score * (1 + 0.3 * early_factor)

如果没有 onset_time，则 root_score = base_root_score。

输出按 root_score 降序排列的 RootCauseCandidate 列表。

十三、Pipeline 实现

在 pipeline.py 中实现：

run_spectral_rca(
    metric_df,
    candidate_edges,
    incident_start_time,
    incident_end_time,
    baseline_start_time,
    baseline_end_time,
    config
)

流程：
1. 调用 MetricAnomalyExpert.detect。
2. 过滤出 candidate_anomaly 和 high_confidence_anomaly。
3. 调用 SpectralGraphRefinementExpert.refine。
4. 调用 RootCauseRankingExpert.rank。
5. 返回：
   - anomaly_evidence
   - edge_evidence
   - refined_graph
   - root_cause_ranking
   - summary_report

十四、反思模块

实现 ReflectionExpert 或简单函数 compare_results。

必须比较：
1. traditional-only anomaly vs spectral-enhanced anomaly；
2. traditional graph refinement vs spectral-enhanced graph refinement；
3. 如果 traditional 检测异常但 spectral 不异常：
   解释为 "level shift without strong dynamic spectral change"。
4. 如果 spectral 检测异常但 traditional 不强：
   解释为 "subtle dynamic anomaly such as periodic oscillation or burst pattern"。
5. 如果某条边时域相关高但频域相似度低：
   标记为 "possible spurious temporal correlation"。
6. 如果某条边频域相似度高但时域滞后不支持方向：
   标记为 "shared pattern but weak directional evidence"。

十五、实验和消融

在 experiments/run_ablation.py 中实现以下开关：

- use_traditional_anomaly=True/False
- use_spectral_anomaly=True/False
- use_lag_corr=True/False
- use_spectral_shape_similarity=True/False
- use_dominant_freq_match=True/False
- use_phase_lag_consistency=True/False
- use_graph_consistency=True/False

至少输出：
1. top-k root cause accuracy，如果有 ground truth；
2. root cause ranking；
3. kept_edges 数量；
4. pruned_edges 数量；
5. 每条边的 evidence；
6. 每个异常 metric 的 traditional_score、spectral_score、final_score。

十六、代码质量要求

1. 所有函数必须有 docstring。
2. 所有阈值必须放在 config 中，不要写死。
3. 避免除以 0，统一使用 eps=1e-8。
4. 对短序列、缺失数据、常数序列必须有保护。
5. 不允许 silent failure，所有异常都要在 evidence 中说明。
6. 输出结果必须能保存为 JSON。
7. 写最小单元测试：
   - 慢变趋势序列应 low_ratio 较高；
   - 周期抖动序列应 dominant_freq_energy_ratio 较高；
   - 尖峰序列应 total_energy 高且频谱较分散；
   - 两条相同频率模式的序列应 spectral_shape_similarity 高；
   - 不同频率模式的序列应 spectral_shape_similarity 低。

十七、优先实现顺序

第一阶段：
1. preprocessing.py
2. anomaly/traditional.py
3. anomaly/spectral.py
4. MetricAnomalyExpert

第二阶段：
5. graph/spectral_edge.py
6. graph/graph_consistency.py
7. SpectralGraphRefinementExpert

第三阶段：
8. RootCauseRankingExpert
9. pipeline.py
10. run_ablation.py
11. tests

十八、注意事项

1. 频域异常检测不替代传统异常检测，而是增强传统异常检测。
2. DFT Total Energy 只表示整体波动强度，不能单独判断故障类型。
3. 故障类型判断必须结合 low_ratio、high_ratio、dominant_freq_energy_ratio 和 spectral_entropy。
4. 频域方法不能单独决定因果方向，方向必须结合候选传播图、时域滞后相关和相位滞后。
5. 标准 Graph Laplacian 更适合无向图，因此第一版不要用它学习方向，只用于图结构一致性检查。
6. 第一版重点证明：频域方法能提升异常 metric 发现质量和候选因果边剪枝质量。