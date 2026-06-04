"""BaseAgent 抽象基类。

技术方案 Agent 通信协议：
- 每个子Agent返回 AgentResult，包含 status(OK/LOW_QUALITY/ERROR)、
  self_quality(0~1)、result(dict)、warnings(list)、suggestion(str)
- Orchestrator通过统一接口处理所有Agent的响应
"""
from __future__ import annotations

import logging
import time
import traceback
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from causalrca_codex.config import AgentLoopConfig
from causalrca_codex.schemas import AgentResult

logger = logging.getLogger("causalrca")


class BaseAgent(ABC):
    """所有Agent的抽象基类，定义统一的生命周期模板方法。

    模板方法 run() 的执行流程：
    1. validate_input() - 检查前置条件（上游层是否已填充）
    2. _execute() - 子类实现的具体计算逻辑
    3. _self_evaluate() - 自评估输出质量 (0~1)
    4. _suggest_next_action() - 根据质量建议下一步操作
    5. 包装为 AgentResult 返回

    错误处理：异常被捕获并包装为 AgentResult(status=ERROR)，
    不会向上传播，Orchestrator可通过status判断并触发恢复。
    """

    name = "BaseAgent"
    purpose = ""
    estimated_cost = "low"
    preconditions: List[str] = []
    produces: List[str] = []
    tunable_params: Dict[str, Any] = {}

    def __init__(self, config: AgentLoopConfig) -> None:
        self.config = config

    @property
    def capabilities(self) -> Dict[str, Any]:
        """返回Agent能力描述，用于Orchestrator决策。"""
        return {
            "name": self.name,
            "purpose": self.purpose,
            "preconditions": self.preconditions,
            "produces": self.produces,
            "tunable_params": self.tunable_params,
            "estimated_cost": self.estimated_cost,
        }

    def run(self, workspace: Dict[str, Any], params: Optional[Dict[str, Any]] = None) -> AgentResult:
        """Agent执行入口（模板方法）。

        包装 _execute() 的完整生命周期，包含：
        - 异常捕获（ERROR状态）
        - 执行计时
        - 自评估和建议生成

        Args:
            workspace: 共享黑板状态
            params: 本次执行的调优参数

        Returns:
            AgentResult: 统一的Agent响应对象
        """
        params = params or {}
        start_time = time.time()

        try:
            # 验证输入
            validation_warnings = self.validate_input(workspace, params)
            if validation_warnings:
                logger.warning(f"[{self.name}] 输入验证警告: {validation_warnings}")

            # 执行核心逻辑
            result = self._execute(workspace, params)

            # 自评估
            quality, warnings = self._self_evaluate(result, workspace, params)
            warnings = list(warnings) + list(validation_warnings)

            elapsed = time.time() - start_time
            logger.info(f"[{self.name}] 完成 质量={quality:.2f} 耗时={elapsed:.2f}s")

            return AgentResult(
                agent_name=self.name,
                status="OK",
                result=result,
                self_assessed_quality=quality,
                warnings=warnings,
                suggested_next_action=self._suggest_next_action(result, quality, warnings),
            )

        except Exception as e:
            elapsed = time.time() - start_time
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"[{self.name}] 执行失败 ({elapsed:.2f}s): {error_msg}")
            logger.debug(f"[{self.name}] 详细堆栈:\n{traceback.format_exc()}")

            return AgentResult(
                agent_name=self.name,
                status="ERROR",
                result={"error": error_msg, "traceback": traceback.format_exc()},
                self_assessed_quality=0.0,
                warnings=[f"Agent execution failed: {error_msg}"],
                suggested_next_action="recover",
            )

    def validate_input(self, workspace: Dict[str, Any], params: Dict[str, Any]) -> List[str]:
        """验证上游依赖层是否已填充。

        Returns:
            警告信息列表（空表示验证通过）
        """
        warnings = []
        for precond in self.preconditions:
            parts = precond.split(".", 1)
            if len(parts) == 2:
                layer, key = parts
                layer_data = workspace.get(layer, {})
                if not layer_data.get(key):
                    warnings.append(f"前置条件未满足: {precond}")
        return warnings

    @abstractmethod
    def _execute(self, workspace: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """子类必须实现的核心执行逻辑。

        Args:
            workspace: 共享黑板状态
            params: 调优参数

        Returns:
            结果字典，将被写入workspace的对应层
        """
        raise NotImplementedError

    def _self_evaluate(
        self,
        result: Dict[str, Any],
        workspace: Dict[str, Any],
        params: Dict[str, Any],
    ) -> Tuple[float, List[str]]:
        """自评估输出质量。

        Args:
            result: _execute()的返回结果
            workspace: 当前workspace状态
            params: 本次执行参数

        Returns:
            (quality, warnings): quality为0~1的质量分数，warnings为警告列表
        """
        return 1.0, []

    def _suggest_next_action(self, result: Dict[str, Any], quality: float, warnings: List[str]) -> str:
        """根据质量评估建议下一步操作。

        Args:
            result: 执行结果
            quality: 自评估质量分数
            warnings: 警告列表

        Returns:
            "continue" 或 "recover"
        """
        if quality < self.config.low_quality_threshold:
            return "recover"
        return "continue"
