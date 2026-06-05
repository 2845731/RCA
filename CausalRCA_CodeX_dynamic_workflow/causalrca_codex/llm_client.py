"""真正的LLM客户端 - 接入大模型API进行推理。

本模块提供与大模型的交互能力，用于：
1. Orchestrator的智能调度决策
2. Agent间的质疑和辩论
3. 因果推理和证据分析
4. 最终根因验证

API配置来自 agent_api.py 文件。
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Optional

# 强制不走代理
os.environ["no_proxy"] = "localhost,127.0.0.1,0.0.0.0"


class LLMClient:
    """真正的大模型客户端，用于多智能体推理。"""

    def __init__(
        self,
        base_url: str = "https://token-plan-cn.xiaomimimo.com/v1",
        api_key: str = "YOUR_API_KEY_HERE",
        model: str = "mimo-v2.5-pro",
        temperature: float = 0.3,
        max_retries: int = 3,
        timeout: int = 60,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout
        self._client = None

    def _get_client(self):
        """延迟初始化OpenAI客户端。"""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout,
            )
        return self._client

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: int = 2000,
    ) -> str:
        """发送聊天请求到大模型，返回文本响应。"""
        client = self._get_client()
        temp = temperature if temperature is not None else self.temperature

        for attempt in range(self.max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=max_tokens,
                )
                content = response.choices[0].message.content
                if content:
                    return content.strip()
            except Exception as e:
                if attempt == self.max_retries - 1:
                    print(f"    [LLM] API调用失败: {type(e).__name__}: {e}")
                    return ""
                time.sleep(1 * (attempt + 1))
        return ""

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: int = 2000,
    ) -> Optional[Dict[str, Any]]:
        """发送请求并解析JSON响应。"""
        response = self.chat(messages, temperature, max_tokens)
        if not response:
            return None

        # 尝试从响应中提取JSON
        # 1. 先尝试直接解析
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # 2. 尝试从markdown代码块中提取
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # 3. 尝试找第一个{到最后一个}
        start = response.find("{")
        end = response.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(response[start:end + 1])
            except json.JSONDecodeError:
                pass

        return None

    def analyze_root_cause(
        self,
        instruction: str,
        anomaly_components: List[Dict[str, Any]],
        causal_graph_info: str,
        candidate_reasons: List[str],
        telemetry_summary: str,
    ) -> Dict[str, Any]:
        """让LLM分析根因，用于多智能体辩论和验证。"""
        messages = [
            {
                "role": "system",
                "content": """你是一个专业的AIOps根因分析专家。你的任务是分析微服务系统的故障根因。

你需要：
1. 分析异常组件的因果关系
2. 区分真正的根因和下游影响
3. 从候选原因中选择最匹配的
4. 给出置信度和推理过程

请用JSON格式输出：
{
    "root_cause_component": "组件名",
    "root_cause_reason": "从候选列表中选择",
    "confidence": 0.0-1.0,
    "reasoning": "详细推理过程",
    "alternative_candidates": ["其他可能的候选"]
}"""
            },
            {
                "role": "user",
                "content": f"""## 故障描述
{instruction}

## 异常组件及其严重度
{json.dumps(anomaly_components, ensure_ascii=False, indent=2)}

## 因果图信息
{causal_graph_info}

## 候选原因列表
{candidate_reasons}

## 遥测数据摘要
{telemetry_summary}

请分析最可能的根因组件和原因。"""
            }
        ]

        result = self.chat_json(messages)
        if result and "root_cause_component" in result:
            return result

        # 返回默认值
        return {
            "root_cause_component": anomaly_components[0]["component"] if anomaly_components else "",
            "root_cause_reason": candidate_reasons[0] if candidate_reasons else "",
            "confidence": 0.5,
            "reasoning": "LLM分析失败，使用默认值",
            "alternative_candidates": []
        }

    def debate_root_cause(
        self,
        hypothesis: Dict[str, Any],
        evidence_for: List[str],
        evidence_against: List[str],
        alternative_view: Dict[str, Any],
    ) -> Dict[str, Any]:
        """让LLM对两个不同的根因假设进行辩论和仲裁。"""
        messages = [
            {
                "role": "system",
                "content": """你是一个根因分析仲裁专家。有两个不同的根因假设需要你评判。

你需要：
1. 评估每个假设的证据强度
2. 找出每个假设的弱点
3. 给出最终判断和理由

请用JSON格式输出：
{
    "winner": "hypothesis_1 或 hypothesis_2 或 inconclusive",
    "confidence": 0.0-1.0,
    "reasoning": "详细评判过程",
    "key_evidence": "决定性证据"
}"""
            },
            {
                "role": "user",
                "content": f"""## 假设1: {hypothesis.get('root_cause_component', '未知')} - {hypothesis.get('root_cause_reason', '未知')}
置信度: {hypothesis.get('confidence', 0)}
支持证据:
{chr(10).join(f'- {e}' for e in evidence_for)}

## 假设2: {alternative_view.get('root_cause_component', '未知')} - {alternative_view.get('root_cause_reason', '未知')}
置信度: {alternative_view.get('confidence', 0)}
支持证据:
{chr(10).join(f'- {e}' for e in evidence_against)}

请评判哪个假设更可能是正确的根因。"""
            }
        ]

        result = self.chat_json(messages)
        if result and "winner" in result:
            return result

        return {
            "winner": "inconclusive",
            "confidence": 0.5,
            "reasoning": "LLM仲裁失败",
            "key_evidence": "无法确定"
        }

    def validate_reason(
        self,
        component: str,
        component_type: str,
        anomaly_kpis: List[str],
        log_evidence: str,
        candidate_reasons: List[str],
    ) -> Dict[str, Any]:
        """让LLM验证选择的原因是否合理。"""
        messages = [
            {
                "role": "system",
                "content": """你是一个故障原因验证专家。根据组件类型、异常KPI和日志证据，验证选择的故障原因是否合理。

请用JSON格式输出：
{
    "most_likely_reason": "从候选列表中选择",
    "confidence": 0.0-1.0,
    "reasoning": "为什么这个原因最可能",
    "kpi_evidence": "哪些KPI支持这个判断",
    "log_evidence": "日志中有什么证据"
}"""
            },
            {
                "role": "user",
                "content": f"""## 组件信息
- 组件名: {component}
- 组件类型: {component_type}

## 异常KPI
{chr(10).join(f'- {kpi}' for kpi in anomaly_kpis)}

## 日志证据
{log_evidence[:2000] if log_evidence else "无日志数据"}

## 候选原因
{chr(10).join(f'- {reason}' for reason in candidate_reasons)}

请选择最匹配的故障原因。"""
            }
        ]

        result = self.chat_json(messages)
        if result and "most_likely_reason" in result:
            return result

        return {
            "most_likely_reason": candidate_reasons[0] if candidate_reasons else "",
            "confidence": 0.5,
            "reasoning": "LLM验证失败",
            "kpi_evidence": "",
            "log_evidence": ""
        }


# 全局单例
_global_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """获取全局LLM客户端实例。"""
    global _global_llm_client
    if _global_llm_client is None:
        _global_llm_client = LLMClient()
    return _global_llm_client
