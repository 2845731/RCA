# Skill: 添加 LogTopologyAgent 并接入 CausalGraphAgent

## 目标
实现一个专门从故障时间窗口日志中挖掘调用拓扑的 Agent，并将其合并到 CausalGraphAgent 的主调用图骨架中。

## 适用范围
- `d:\GitHubDownload\OpenRCA\CausalRCA_CodeX\causalrca_codex\`

## 输入数据
- `data_layer.raw_traces_window` (故障窗口内 trace)
- `data_layer.raw_logs` (故障窗口内 log DataFrame)
- `fault_id_layer.refined_candidates` (候选组件)

## 涉及文件

| 文件 | 改动 |
|------|------|
| `causalrca_codex/agents/log_topology_agent.py` | **新建** |
| `causalrca_codex/agents/__init__.py` | 导出 LogTopologyAgent |
| `causalrca_codex/agents/data_agent.py` | 暴露 raw_logs (故障窗口) |
| `causalrca_codex/agents/causal_graph_agent.py` | 接入新 agent, 合并拓扑 |
| `causalrca_codex/orchestrator.py` | 在合适阶段运行 LogTopologyAgent |
| `causalrca_codex/workspace.py` | 新增 `log_topology_layer` |

## 实施步骤

### Step 1: 创建 `log_topology_agent.py`

#### 1.1 类结构
```python
class LogTopologyAgent(BaseAgent):
    name = "LogTopologyAgent"
    purpose = "从故障窗口日志中挖掘 service->db/redis 调用拓扑, 与主调用图合并"
    preconditions = ["data_layer.raw_logs"]
    produces = ["log_topology_layer"]
```

#### 1.2 挖掘策略 (按优先级)

**策略A: DBCP2 datasource 启动日志** (Bank catalina 数据有)
- 模式: `Name = XXX_MYSQL Property ...` 或 `Name = XXX_REDIS`
- 正则: `Name = ([A-Za-z0-9_\-]+) `
- 输出: `{(service, datasource_name)}`

**策略B: Jedis redis 客户端连接**
- 模式: `redis.clients.jedis.JedisSentinelPool.initPool Created JedisPool to master`
- 输出: `{(service, redis_pool)}`

**策略C: JDBC URL (如日志含 jdbc:mysql://)**
- 模式: `jdbc:(mysql|postgresql|oracle)://IP:port/...`
- 输出: `{(service, jdbc_url_host)}`

**策略D: Spring/Hikari 错误日志 (运行期)**
- 模式: `HikariPool.*connection.*(timeout|refused)` 或 `Communications link failure`
- 输出: `{(service, db)}` (弱信号)

#### 1.3 类成员方法
- `_execute(self, workspace, params) -> Dict`
- `_extract_dbcp2_datasources(log_df) -> List[Tuple[cmdb_id, datasource_name]]`
- `_extract_jedis_pools(log_df) -> List[Tuple[cmdb_id, "redis"]]`
- `_extract_jdbc_urls(log_df) -> List[Tuple[cmdb_id, host]]`
- `_datasource_name_to_cmdb(datasource_name, candidate_set) -> Optional[cmdb_id]`
  - 业务名包含 "MYSQL" → 在 refined_candidates 中选第一个 "Mysql0X"
  - 业务名包含 "REDIS" → 选第一个 "Redis0X"
  - 业务名包含 "ORACLE" → 不处理
  - 否则返回 None
- `_print_extraction_results(self, ...)` 详细打印挖掘过程

#### 1.4 输出格式
```python
workspace["log_topology_layer"] = {
    "raw_edges": [                # 原始挖掘的边 (caller, callee, source, weight)
        ("Tomcat01", "Mysql01", "dbcp2:CMBCSA_BPM_HIS_MYSQL", 5),
        ...
    ],
    "call_counts": {(caller, callee): weight, ...},  # 转成 call_counts 格式
    "matched_with_main_graph": [...],   # 与主图重叠的边
    "unmatched_with_main_graph": [...], # 仅 log 拓扑有的边
    "datasource_mapping": {...},        # datasource_name -> cmdb_id 映射细节
    "extraction_stats": {...},          # 各类策略的命中数
}
```

### Step 2: 暴露故障窗口 log (data_agent.py)

检查 `data_agent.py` 是否有 `raw_logs` 字段。如果没有，从 `frames["log"]` 添加故障窗口裁剪后的 `raw_logs_window`。
打印每条 log 涉及的 cmdb_id 数量。

### Step 3: 修改 CausalGraphAgent

在 `causal_graph_agent.py` 的 `_execute` 中：
1. 调用 `LogTopologyAgent` (在构建主调用图后)
2. 拿到 `log_topology_layer.call_counts`
3. 合并策略：
   - 主图已有边: `merged[key] = max(主图, log)` (取较大权重, 体现一致性)
   - 主图无, log 有边: `merged[key] = log_weight` (新增, 视为"配置层拓扑边")
   - 主图有, log 无: 保持原样
4. 打印合并前后对比
5. 标记"日志来源边" `from_log_topology=True`, 后续 r_trace 不应过高

### Step 4: orchestrator.py 集成

在 CausalGraphAgent 之前调用 LogTopologyAgent, 或者让 CausalGraphAgent 内部调用 (推荐后者, 减少编排复杂度)。

### Step 5: 详细打印模板

```
[LogTopologyAgent] ========== 开始日志拓扑挖掘 ==========
[LogTopologyAgent] 故障窗口: [ts_start, ts_end]  (X-Y 时间)
[LogTopologyAgent] log 行数=N, 涉及 cmdb_id: [Tomcat01-04, apache01/02]

[LogTopologyAgent] --- 策略A: DBCP2 datasource 启动日志 ---
[LogTopologyAgent]   命中: N 条
[LogTopologyAgent]   配对详情:
[LogTopologyAgent]     Tomcat01 -> CMBCSA_BPM_HIS_MYSQL (32 次)
[LogTopologyAgent]     Tomcat01 -> CMBCSA_BPM_MYSQL     (32 次)
[LogTopologyAgent]   datasource_name -> cmdb_id 映射:
[LogTopologyAgent]     CMBCSA_BPM_HIS_MYSQL -> Mysql01 (启发式: 含MYSQL+候选集中第一个)
[LogTopologyAgent]     CMBCSA_BPM_MYSQL     -> Mysql02

[LogTopologyAgent] --- 策略B: Jedis redis 连接 ---
[LogTopologyAgent]   命中: M 条
[LogTopologyAgent]   Tomcat01-04 全部启用了 redis 连接池

[LogTopologyAgent] 挖掘结果汇总: X 条 service->db/redis 边
[LogTopologyAgent]   {(Tomcat01, Mysql01), (Tomcat01, Mysql02), (Tomcat01, Redis01), ...}

[LogTopologyAgent] ========== 与主调用图对比 ==========
[LogTopologyAgent] 主调用图 (来自 trace) 节点: {apache01, apache02, Tomcat01-04}
[LogTopologyAgent] log 拓扑节点: {Tomcat01-04, Mysql01, Mysql02, Redis01, Redis02}
[LogTopologyAgent] 重叠节点: {Tomcat01-04}  → ✅ 合并
[LogTopologyAgent] 新增节点 (仅 log 有): {Mysql01, Mysql02, Redis01, Redis02}
[LogTopologyAgent] 主图已有边 vs log 拓扑边 重叠数: 0 (trace 没抓到)
[LogTopologyAgent] 合并后调用对数: A -> B
[LogTopologyAgent]   新增边 (log->main): [Tomcat01->Mysql01, Tomcat01->Mysql02, ...]
[LogTopologyAgent]   共同边 (both): []
[LogTopologyAgent]   独立边 (仅 main): [apache01->Tomcat01, ...]

[LogTopologyAgent] ========== 完成 ==========
```

**无重叠时的失败打印模板**:
```
[LogTopologyAgent] ⚠️ 无重叠节点: 主图节点 = {apache01, ...}, log 拓扑节点 = {其他}
[LogTopologyAgent] 原因: log 拓扑挖掘的节点不在 refined_candidates / 主图节点中
[LogTopologyAgent] 处理: 保留 log_topology_layer 独立图, 不合并
[LogTopologyAgent] 退出状态: FAILED (no overlap)
```

## 关键原则
1. **绝不能臆测全连接**: 严格按挖掘到的边合并, 不自动补全
2. **datasource_name -> cmdb_id 启发式必须保守**: 仅在名字包含 MYSQL/REDIS 且候选集中有对应组件时使用
3. **r_trace 标弱**: 来自 log 的边 r_trace 上限 0.55 (介于推断 0.45 和 trace 1-exp(-c/5) 之间)
4. **失败也要打印**: 哪怕没挖到任何边, 也要打印"无 DBCP2 启动日志 / 无 jedis 连接 / 无 jdbc URL"
5. **不修改 CausalGraphAgent 现有 score 公式**: 仅在 r_trace 之外, 给 log 拓扑边一个独立标识

## 验证步骤
1. 运行 `python -c "import ast; ast.parse(open('causalrca_codex/agents/log_topology_agent.py', encoding='utf-8').read())"`
2. 运行 `python -c "import causalrca_codex.agents.log_topology_agent"` 确认 import 正常
3. 跑 Bank row_0 看输出

## 不要做的事
- 不要让 agent 改 Agent 4 (CausalGraphAgent) 的 Scheme-F 公式
- 不要从 component name 前缀猜 db (Mysql0X -> db), 已弃用
- 不要把 log 拓扑边与 trace 边同等对待 (必须降权)
- 不要在 orchestrator 阶段重复执行
