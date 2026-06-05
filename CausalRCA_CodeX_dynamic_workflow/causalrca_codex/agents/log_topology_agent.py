"""Agent 4.5: LogTopologyAgent
====================================

目的: 从故障时间窗口内的 log 数据中挖掘 service->db/redis 调用拓扑,
      与 CausalGraphAgent 的主调用图(来自 trace) 合并。

合并策略:
  1. 若 log 拓扑节点与主调用图节点有重叠 (>= 1 个重叠 component),
     则将 log 拓扑边合并到主调用图中 (call_counts 累加权重)
  2. 若无重叠, 则保留为独立 log_topology_layer, 标记 FAILED
  3. log 拓扑边的 r_trace 上限 = 0.55 (介于推断 0.45 和 trace 1.0 之间)

挖掘策略 (按优先级):
  A) DBCP2 datasource 启动日志 (catalina): Name = XXX_MYSQL
  B) Jedis redis 客户端连接: Created JedisPool to master
  C) JDBC URL: jdbc:mysql/postgresql/oracle://host:port
  D) Spring/Hikari 错误日志 (运行期): connection timeout / Communications link failure

输出: workspace["log_topology_layer"]
  - call_counts: {(caller, callee): weight}
  - matched_with_main: [(caller, callee), ...]
  - unmatched_with_main: [(caller, callee), ...]
  - extraction_stats: {strategy_A: N, strategy_B: M, ...}
  - status: "MERGED" / "INDEPENDENT" / "NO_LOG"
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from causalrca_codex.agents.base import BaseAgent
from causalrca_codex.schemas import AgentResult


# 来自 log 的拓扑边权重上限 (介于推断 0.45 和 trace 真实边之间)
LOG_TOPOLOGY_MAX_R_TRACE = 0.55
# log 拓扑边的 call_count (用于 trace_reliability 公式: 1-exp(-c/5))
LOG_TOPOLOGY_CALL_COUNT = 5


class LogTopologyAgent(BaseAgent):
    """从故障窗口日志中挖掘 service->db/redis 调用拓扑。"""

    name = "LogTopologyAgent"
    purpose = "从故障窗口日志挖掘调用拓扑, 与 CausalGraphAgent 主图合并"
    preconditions = ["data_layer.raw_logs"]
    produces = ["log_topology_layer"]
    estimated_cost = "low"

    def _execute(self, workspace: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        query = workspace["task"]["query"]
        log_frames = workspace["data_layer"].get("raw_logs", [])
        refined = list(workspace["fault_id_layer"].get("refined_candidates", []))
        anomalous = list(workspace["association_layer"].get("candidate_set", []))

        # 1) 加载故障窗口内 log
        if not log_frames:
            self._print_no_log(query)
            return self._make_result("NO_LOG", {}, [], [], {}, reason="data_layer.raw_logs 为空")

        log_df = self._concat_log_frames(log_frames)
        if log_df is None or log_df.empty:
            self._print_no_log(query)
            return self._make_result("NO_LOG", {}, [], [], {}, reason="合并后 log 为空")

        print(f"    [LogTopologyAgent] ========== 开始日志拓扑挖掘 ==========")
        tw = workspace["data_layer"].get("trace_time_windows", {})
        if tw:
            print(f"    [LogTopologyAgent] 故障窗口: [{tw.get('effective_start_ts', '?')}, {tw.get('end_ts', '?')}]")
        print(f"    [LogTopologyAgent] log 行数={len(log_df)}, "
              f"涉及 cmdb_id={sorted(log_df['cmdb_id'].dropna().unique().tolist()) if 'cmdb_id' in log_df.columns else 'N/A'}")

        candidate_set = list(set(refined) | set(anomalous))

        # 2) 各种策略挖掘
        stats: Dict[str, int] = {}

        # 策略A: DBCP2 datasource
        ds_pairs = self._extract_dbcp2_datasources(log_df)
        stats["A_DBCP2"] = len(ds_pairs)
        print(f"    [LogTopologyAgent] --- 策略A: DBCP2 datasource 启动日志 ---")
        print(f"    [LogTopologyAgent]   命中 {len(ds_pairs)} 条 (cmdb_id, datasource_name) 配对")
        for cmdb, dsn in ds_pairs[:10]:
            print(f"    [LogTopologyAgent]     {cmdb} -> datasource='{dsn}'")
        if len(ds_pairs) > 10:
            print(f"    [LogTopologyAgent]     ... 另有 {len(ds_pairs) - 10} 条省略")

        # 策略B: Jedis redis
        jedis_pairs = self._extract_jedis_pools(log_df)
        stats["B_JEDIS"] = len(jedis_pairs)
        print(f"    [LogTopologyAgent] --- 策略B: Jedis redis 连接池 ---")
        print(f"    [LogTopologyAgent]   命中 {len(jedis_pairs)} 条 (cmdb_id) 配对")
        for cmdb in sorted(set(p[0] for p in jedis_pairs))[:10]:
            print(f"    [LogTopologyAgent]     {cmdb} -> redis 池 (启动时)")

        # 策略C: JDBC URL
        jdbc_pairs = self._extract_jdbc_urls(log_df)
        stats["C_JDBC_URL"] = len(jdbc_pairs)
        print(f"    [LogTopologyAgent] --- 策略C: JDBC URL ---")
        print(f"    [LogTopologyAgent]   命中 {len(jdbc_pairs)} 条 (cmdb_id, host) 配对")
        for cmdb, host in jdbc_pairs[:5]:
            print(f"    [LogTopologyAgent]     {cmdb} -> jdbc://{host}")

        # 策略D: Spring/Hikari 错误日志
        err_pairs = self._extract_connection_errors(log_df)
        stats["D_CONN_ERROR"] = len(err_pairs)
        print(f"    [LogTopologyAgent] --- 策略D: 连接错误日志 (运行期) ---")
        print(f"    [LogTopologyAgent]   命中 {len(err_pairs)} 条 (cmdb_id) 配对")

        # 策略E: access_log 调用方识别 (Bank 数据集关键策略)
        access_pairs = self._extract_access_log_callers(log_df)
        stats["E_ACCESS_LOG"] = len(access_pairs)
        print(f"    [LogTopologyAgent] --- 策略E: access_log 调用方识别 ---")
        print(f"    [LogTopologyAgent]   命中 {len(access_pairs)} 条 (caller, callee) 配对")
        # 统计 caller->callee
        from collections import Counter
        access_counter = Counter(access_pairs)
        for (caller, callee), cnt in access_counter.most_common(10):
            print(f"    [LogTopologyAgent]     {caller} -> {callee}  (count={cnt})")

        # 3) 汇总, 解析 datasource_name -> cmdb_id 映射
        print(f"    [LogTopologyAgent] ========== 启发式映射 datasource_name -> cmdb_id ==========")
        ds_mapping = self._build_datasource_mapping(ds_pairs, candidate_set)
        if ds_mapping:
            for dsn, cmdb in ds_mapping.items():
                print(f"    [LogTopologyAgent]   '{dsn}' -> {cmdb}")
        else:
            print(f"    [LogTopologyAgent]   (无映射, 因为 datasource 名字里不含 MYSQL/REDIS 等关键字)")

        # 4) 转换成 call_counts
        log_call_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        raw_edges: List[Tuple[str, str, str, int]] = []

        # 4.1) DBCP2 配对转边
        for cmdb_id, dsn in ds_pairs:
            target = ds_mapping.get(dsn)
            if target is None:
                continue
            log_call_counts[(cmdb_id, target)] += LOG_TOPOLOGY_CALL_COUNT
            raw_edges.append((cmdb_id, target, f"A_DBCP2:{dsn}", LOG_TOPOLOGY_CALL_COUNT))

        # 4.2) Jedis 配对转边 (连所有 redis 候选)
        redis_candidates = [c for c in candidate_set if "redis" in c.lower() or "Redis" in c]
        jedis_cms = sorted(set(p[0] for p in jedis_pairs))
        for cmdb_id in jedis_cms:
            for r in redis_candidates:
                log_call_counts[(cmdb_id, r)] += LOG_TOPOLOGY_CALL_COUNT
                raw_edges.append((cmdb_id, r, "B_JEDIS:pool_init", LOG_TOPOLOGY_CALL_COUNT))

        # 4.3) JDBC URL 配对转边
        for cmdb_id, host in jdbc_pairs:
            target = self._jdbc_host_to_cmdb(host, candidate_set)
            if target is None:
                continue
            log_call_counts[(cmdb_id, target)] += LOG_TOPOLOGY_CALL_COUNT
            raw_edges.append((cmdb_id, target, f"C_JDBC_URL:{host}", LOG_TOPOLOGY_CALL_COUNT))

        # 4.4) 运行期错误作为边 (弱信号, 权重低)
        err_cms = sorted(set(p[0] for p in err_pairs))
        # 错误日志中提到 db 时, 表示 service 调用 db 失败
        # 简化: 不强行建边, 仅记录为信号
        if err_cms:
            print(f"    [LogTopologyAgent]   运行期错误涉及的 cmdb_id (作为弱信号记录): {err_cms}")

        # 4.5) access_log 调用方识别 (Bank 关键: IG02 -> Tomcat01 等)
        # 已是 (caller, callee) 二元组, 直接累加
        for caller, callee in access_pairs:
            log_call_counts[(caller, callee)] += 1
            raw_edges.append((caller, callee, "E_ACCESS_LOG:http_caller", 1))
        # 输出汇总时, 同时打印 access_log 贡献最多的边
        if access_counter:
            print(f"    [LogTopologyAgent]   access_log Top 边 (按出现频次):")
            for (caller, callee), cnt in access_counter.most_common(15):
                print(f"    [LogTopologyAgent]     {caller} -> {callee}  (count={cnt})")

        print(f"    [LogTopologyAgent] 挖掘结果汇总: {len(log_call_counts)} 条 service->db/redis 边")
        if log_call_counts:
            for (s, t), c in sorted(log_call_counts.items()):
                print(f"    [LogTopologyAgent]   {s} -> {t}  (call_count={c})")

        # 5) 与主调用图 (如果存在) 对比
        existing_main = self._get_existing_main_call_counts(workspace)

        print(f"    [LogTopologyAgent] ========== 与主调用图对比 ==========")
        print(f"    [LogTopologyAgent] 主调用图边数: {len(existing_main)}")
        print(f"    [LogTopologyAgent] log 拓扑边数: {len(log_call_counts)}")

        main_nodes = set()
        for s, t in existing_main.keys():
            main_nodes.add(s)
            main_nodes.add(t)
        log_nodes = set()
        for s, t in log_call_counts.keys():
            log_nodes.add(s)
            log_nodes.add(t)

        overlap_nodes = main_nodes & log_nodes
        print(f"    [LogTopologyAgent] 主图节点: {sorted(main_nodes)}")
        print(f"    [LogTopologyAgent] log 拓扑节点: {sorted(log_nodes)}")
        print(f"    [LogTopologyAgent] 重叠节点: {sorted(overlap_nodes)}")

        matched = []
        unmatched = []
        for k, v in log_call_counts.items():
            if k in existing_main:
                matched.append(k)
            else:
                unmatched.append(k)
        print(f"    [LogTopologyAgent] 主图 & log 拓扑 都有的边: {len(matched)}")
        print(f"    [LogTopologyAgent] 仅 log 拓扑有的边: {len(unmatched)}")

        # 6) 决策: 合并 / 独立
        status = self._decide_status(overlap_nodes, log_call_counts, main_nodes, log_nodes)
        if status == "MERGED":
            print(f"    [LogTopologyAgent] [OK] 有重叠节点 ({len(overlap_nodes)} 个), 决策: 合并到主图")
            for k in unmatched:
                print(f"    [LogTopologyAgent]   新增边 (log->main): {k[0]} -> {k[1]}")
        elif status == "INDEPENDENT":
            print(f"    [LogTopologyAgent] [WARN] 无重叠节点, 决策: 保留为独立图")
            print(f"    [LogTopologyAgent]   主图节点 = {sorted(main_nodes)}")
            print(f"    [LogTopologyAgent]   log 节点  = {sorted(log_nodes)}")
            print(f"    [LogTopologyAgent]   退出状态: FAILED (no overlap)")
        else:
            print(f"    [LogTopologyAgent] [X] 状态: {status}")

        print(f"    [LogTopologyAgent] ========== 完成 ==========")

        # 7) 写回 workspace
        layer = {
            "raw_edges": raw_edges,
            "call_counts": dict(log_call_counts),
            "matched_with_main": [f"{k[0]}->{k[1]}" for k in matched],
            "unmatched_with_main": [f"{k[0]}->{k[1]}" for k in unmatched],
            "overlap_nodes": sorted(overlap_nodes),
            "log_only_nodes": sorted(log_nodes - main_nodes),
            "main_only_nodes": sorted(main_nodes - log_nodes),
            "datasource_mapping": ds_mapping,
            "extraction_stats": stats,
            "status": status,
            "max_r_trace": LOG_TOPOLOGY_MAX_R_TRACE,
            "log_call_count": LOG_TOPOLOGY_CALL_COUNT,
        }
        workspace.setdefault("log_topology_layer", {}).update(layer)

        return {
            "log_topology_layer": layer,
        }

    # ============================================================
    # 各类策略的实现
    # ============================================================

    @staticmethod
    def _concat_log_frames(frames) -> Optional[pd.DataFrame]:
        """合并 log 帧: 支持 TelemetryFrame (有 .data) 或纯 DataFrame."""
        if not frames:
            return None
        parts = []
        for f in frames:
            if hasattr(f, "rows") and getattr(f, "rows", 0) == 0:
                continue
            if hasattr(f, "data"):  # TelemetryFrame
                df = f.data
            elif hasattr(f, "df"):
                df = f.df
            else:
                df = f
            if df is None or len(df) == 0:
                continue
            parts.append(df)
        if not parts:
            return None
        return pd.concat(parts, ignore_index=True)

    @staticmethod
    def _extract_dbcp2_datasources(log_df: pd.DataFrame) -> List[Tuple[str, str]]:
        """策略A: DBCP2 datasource 启动日志.

        模式: 'Name = XXX_MYSQL Property ...' / 'Name = XXX_REDIS Ignoring ...'
        """
        if "value" not in log_df.columns or "cmdb_id" not in log_df.columns:
            return []
        # 抓 Name = XXX_DATASOURCE_NAME 后跟 Property|Ignoring|开头
        pattern = re.compile(r"Name\s*=\s*([A-Za-z0-9_\-]+?)(?:\s+(?:Property|Ignoring))")
        out = []
        sub = log_df[log_df["value"].astype(str).str.contains("Name\s*=", na=False)]
        for _, r in sub.iterrows():
            v = str(r["value"])
            m = pattern.search(v)
            if not m:
                continue
            dsn = m.group(1)
            # 排除明显非 db/redis 的 (例如 'Host')
            if dsn.upper() in {"HOST", "USER", "PASSWORD", "URL", "DRIVER"}:
                continue
            out.append((str(r["cmdb_id"]), dsn))
        return out

    @staticmethod
    def _extract_jedis_pools(log_df: pd.DataFrame) -> List[Tuple[str, str]]:
        """策略B: Jedis redis 客户端连接池.

        模式: 'Created JedisPool to master at IPAddress:Port' 或 'initPool'
        """
        if "value" not in log_df.columns or "cmdb_id" not in log_df.columns:
            return []
        mask = log_df["value"].astype(str).str.contains(
            r"JedisPool|redis\.clients\.jedis", case=False, regex=True, na=False
        )
        sub = log_df[mask]
        return [(str(r["cmdb_id"]), "redis") for _, r in sub.iterrows()]

    @staticmethod
    def _extract_jdbc_urls(log_df: pd.DataFrame) -> List[Tuple[str, str]]:
        """策略C: JDBC URL.

        模式: 'jdbc:(mysql|postgresql|oracle)://IP:port/db'
        """
        if "value" not in log_df.columns or "cmdb_id" not in log_df.columns:
            return []
        pattern = re.compile(r"jdbc:(?:mysql|postgresql|oracle)://([^/:\s]+):?(\d+)?")
        out = []
        sub = log_df[log_df["value"].astype(str).str.contains("jdbc:", na=False)]
        for _, r in sub.iterrows():
            v = str(r["value"])
            m = pattern.search(v)
            if not m:
                continue
            host = m.group(1)
            port = m.group(2) or ""
            out.append((str(r["cmdb_id"]), f"{host}:{port}" if port else host))
        return out

    @staticmethod
    def _extract_access_log_callers(log_df: pd.DataFrame) -> List[Tuple[str, str]]:
        """策略E: access_log 调用方识别.

        模式: 'IG02 POST /UOCP/...' / 'IG01 GET /...' / 'apache01 ...'
        - log 行 cmdb_id = 被调用方 (Tomcat01 之类, 实际处理请求的 service)
        - value 开头第一个 token (如 IG02, IG01, MG01, apache01) = 调用方
        用途: 当 trace 抓不到 service->service 边时, 用 log access_log 补
        """
        out: List[Tuple[str, str]] = []
        if "value" not in log_df.columns or "cmdb_id" not in log_df.columns:
            return out
        # 优先看 log_name 列 (若有)
        if "log_name" in log_df.columns:
            mask_name = log_df["log_name"].astype(str).str.contains(
                r"access_log|localhost_access_log", case=False, regex=True, na=False
            )
            sub = log_df[mask_name]
        else:
            sub = log_df
        # value 开头匹配 'XXX METHOD /...' 格式
        pattern = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_\-]+?)\s+(?:GET|POST|PUT|DELETE|HEAD|PATCH)\s+\S+", re.M)
        for _, r in sub.iterrows():
            v = str(r["value"])
            m = pattern.search(v)
            if not m:
                continue
            caller = m.group(1)
            callee = str(r["cmdb_id"])
            # 排除明显非组件 (HTTP/0.29.0, k6/0.29.0, Mozilla 等)
            if caller.lower() in {"http", "https", "k6", "curl", "wget", "mozilla"}:
                continue
            if caller == callee:
                continue
            out.append((caller, callee))
        return out

    @staticmethod
    def _extract_connection_errors(log_df: pd.DataFrame) -> List[Tuple[str, str]]:
        """策略D: 运行期连接错误 (弱信号)."""
        if "value" not in log_df.columns or "cmdb_id" not in log_df.columns:
            return []
        mask = log_df["value"].astype(str).str.contains(
            r"Communications link failure|connection.{0,5}timeout|connection.{0,5}refused|"
            r"HikariPool.*timeout|connection.{0,5}closed",
            case=False, regex=True, na=False,
        )
        sub = log_df[mask]
        return [(str(r["cmdb_id"]), "conn_error") for _, r in sub.iterrows()]

    @staticmethod
    def _build_datasource_mapping(
        ds_pairs: List[Tuple[str, str]], candidate_set: List[str]
    ) -> Dict[str, str]:
        """将 datasource_name 启发式映射到 cmdb_id.

        规则:
          1) 名字含 MYSQL/MySQL -> 选 candidate_set 中的第一个 Mysql0X
          2) 名字含 REDIS/Redis -> 选 candidate_set 中的第一个 Redis0X
          3) 名字含 ORACLE/Oracle -> 不处理
          4) 否则不处理
        """
        if not ds_pairs:
            return {}
        mysql_cands = [c for c in candidate_set if re.search(r"mysql", c, re.IGNORECASE)]
        redis_cands = [c for c in candidate_set if re.search(r"redis", c, re.IGNORECASE)]
        oracle_cands = [c for c in candidate_set if re.search(r"oracle", c, re.IGNORECASE)]
        db_cands = [c for c in candidate_set if re.search(r"db_\d+", c, re.IGNORECASE)]

        mapping: Dict[str, str] = {}
        # 用 set 去重, 避免重复计算
        unique_dsn = sorted(set(dsn for _, dsn in ds_pairs))
        for dsn in unique_dsn:
            target = None
            if re.search(r"mysql", dsn, re.IGNORECASE):
                if mysql_cands:
                    target = mysql_cands[0]
            elif re.search(r"redis", dsn, re.IGNORECASE):
                if redis_cands:
                    target = redis_cands[0]
            elif re.search(r"oracle", dsn, re.IGNORECASE):
                if oracle_cands:
                    target = oracle_cands[0]
            elif db_cands:
                # db_xxx 形式 (Telecom)
                target = db_cands[0]
            if target is not None:
                mapping[dsn] = target
        return mapping

    @staticmethod
    def _jdbc_host_to_cmdb(host: str, candidate_set: List[str]) -> Optional[str]:
        """JDBC host 启发式映射到 cmdb_id.

        简化: 仅当 host 是 IP 形式且 candidate_set 包含 Mysql0X/Redis0X 时返回第一个匹配.
        """
        mysql_cands = [c for c in candidate_set if re.search(r"mysql", c, re.IGNORECASE)]
        redis_cands = [c for c in candidate_set if re.search(r"redis", c, re.IGNORECASE)]
        if mysql_cands and re.search(r"\d+\.\d+\.\d+\.\d+", host):
            return mysql_cands[0]
        if redis_cands and re.search(r"\d+\.\d+\.\d+\.\d+", host):
            return redis_cands[0]
        return None

    @staticmethod
    def _get_existing_main_call_counts(workspace: Dict[str, Any]) -> Dict[Tuple[str, str], int]:
        """读取 CausalGraphAgent 写下的主调用图.

        在 CausalGraphAgent 中, 主调用图保存在:
          causal_graph_layer.local_call_graph: {f"{caller}->{callee}": count}
        或者:
          data_layer.local_call_graph: 同上 (兼容)
        """
        for key in ("causal_graph_layer", "data_layer"):
            layer = workspace.get(key, {})
            lcg = layer.get("local_call_graph")
            if not lcg:
                continue
            out: Dict[Tuple[str, str], int] = {}
            for k, v in lcg.items():
                if "->" not in k:
                    continue
                a, b = k.split("->", 1)
                out[(a, b)] = int(v)
            return out
        return {}

    @staticmethod
    def _decide_status(
        overlap_nodes: Set[str],
        log_call_counts: Dict[Tuple[str, str], int],
        main_nodes: Set[str],
        log_nodes: Set[str],
    ) -> str:
        if not log_call_counts:
            return "EMPTY"
        if overlap_nodes:
            return "MERGED"
        return "INDEPENDENT"

    @staticmethod
    def _print_no_log(query) -> None:
        print(f"    [LogTopologyAgent] [WARN] 无 log 数据 (dataset={query.dataset})")
        print(f"    [LogTopologyAgent]   跳过日志拓扑提取")

    @staticmethod
    def _make_result(
        status: str,
        call_counts: Dict[Tuple[str, str], int],
        matched: List,
        unmatched: List,
        stats: Dict[str, int],
        reason: str = "",
    ) -> Dict[str, Any]:
        layer = {
            "raw_edges": [],
            "call_counts": dict(call_counts),
            "matched_with_main": [f"{k[0]}->{k[1]}" for k in matched],
            "unmatched_with_main": [f"{k[0]}->{k[1]}" for k in unmatched],
            "overlap_nodes": [],
            "log_only_nodes": [],
            "main_only_nodes": [],
            "datasource_mapping": {},
            "extraction_stats": stats,
            "status": status,
            "max_r_trace": LOG_TOPOLOGY_MAX_R_TRACE,
            "log_call_count": LOG_TOPOLOGY_CALL_COUNT,
            "reason": reason,
        }
        return {"log_topology_layer": layer}
