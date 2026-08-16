"""LangChain v1 Single-Agent Candidate Runner。"""

import logging
from collections.abc import Mapping, Sequence
from typing import Any, Callable, Protocol

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from app.agent.agent_prompt import build_agent_tool_calling_system_prompt
from app.agent.context import ToolExecutionContext
from app.agent.frameworks.langchain.run_observer_bridge import (
    LangChainRunObserverBridge,
)
from app.agent.frameworks.langchain.tool_adapter import LangChainToolAdapter
from app.agent.run_observer import AgentRunObserver
from app.agent.tools.base import BaseAgentTool


logger = logging.getLogger(__name__)


class LangChainAgentGraph(Protocol):
    """A2 Runner 真正依赖的最小 LangChain Graph 接口。"""

    def invoke(
        self,
        input: dict[str, Any],
        config: dict[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        """执行一次同步 Agent Graph。"""
        ...


class LangChainAgentError(RuntimeError):
    """Framework Candidate Runner 可识别的基础异常。"""

    code = "langchain_agent_error"


class LangChainAgentLimitError(LangChainAgentError):
    """LangChain Graph 达到 recursion limit。"""

    code = "langchain_recursion_limit"


class LangChainAgentResult(BaseModel):
    """一次 LangChain Candidate Run 的稳定结果，字段对齐 NativeAgentResult。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    answer: str = Field(min_length=1)
    turns: int = Field(ge=1)
    tool_call_count: int = Field(ge=0)


class LangChainSingleAgentRunner:
    """v2.1-A2 LangChain Single-Agent Candidate Runner。

    负责：
    - 使用 LangChain v1 create_agent 构建一次请求级 Agent Graph；
    - 通过 LangChainToolAdapter 绑定现有 Tool 与可信执行上下文；
    - 解析最终回答与最小运行计数，供后续 Native / LangChain 对照。

    当前刻意不负责：
    - 替换生产 /agent/chat；
    - AgentRun/ToolCall 生命周期持久化；
    - SSE；
    - 与 Native 完全等价的 repeated-call / tool budget / run timeout。

    后三项会在 Framework Candidate 通过最小闭环后再逐层补齐，避免
    一开始就把 LangChain 内部机制和现有 Runtime 约束混在一起。
    """

    RUNNER_VERSION = "1.0.0"
    AGENT_NAME = "knowledge_assistant_langchain_candidate"

    def __init__(
        self,
        *,
        model: Any,
        tools: Sequence[BaseAgentTool[Any, Any]],
        recursion_limit: int = 12,
        agent_factory: Callable[..., LangChainAgentGraph] | None = None,
    ) -> None:
        if not tools:
            raise ValueError("tools cannot be empty")
        if recursion_limit <= 0:
            raise ValueError("recursion_limit must be greater than 0")

        self.model = model
        self.tools = list(tools)
        self.recursion_limit = recursion_limit
        self._tool_adapter = LangChainToolAdapter(self.tools)
        self._agent_factory = agent_factory

    def run(
        self,
        *,
        db: Session,
        context: ToolExecutionContext,
        message: str,
        observer: AgentRunObserver | None = None,
    ) -> LangChainAgentResult:
        """执行一次同步 LangChain Candidate Run。"""

        normalized_message = self._normalize_message(message)
        bound_tools = self._tool_adapter.bind_tools(
            db=db,
            context=context,
        )
        agent_factory = self._agent_factory or self._load_create_agent()

        agent_kwargs: dict[str, Any] = {
            "model": self.model,
            "tools": bound_tools,
            "system_prompt": build_agent_tool_calling_system_prompt(),
            "name": self.AGENT_NAME,
        }
        if observer is not None:
            agent_kwargs["middleware"] = [
                LangChainRunObserverBridge(observer).build_middleware()
            ]

        graph = agent_factory(**agent_kwargs)

        logger.info(
            "LangChain agent run started: request_id=%s tool_count=%d "
            "recursion_limit=%d",
            context.request_id,
            len(bound_tools),
            self.recursion_limit,
        )

        try:
            state = graph.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": normalized_message,
                        }
                    ]
                },
                config={
                    "recursion_limit": self.recursion_limit,
                    "max_concurrency": 1,
                },
            )
        except RecursionError as exc:
            raise LangChainAgentLimitError(
                "langchain agent exceeded recursion_limit"
            ) from exc

        messages = self._extract_messages(state)
        answer = self._extract_final_answer(messages)
        turns = sum(1 for item in messages if self._is_ai_message(item))
        tool_call_count = sum(
            len(self._extract_tool_calls(item))
            for item in messages
            if self._is_ai_message(item)
        )

        if observer is not None:
            observer.on_final_answer(answer)

        if turns <= 0:
            raise LangChainAgentError(
                "langchain agent completed without model response"
            )

        logger.info(
            "LangChain agent run completed: request_id=%s turns=%d "
            "tool_call_count=%d",
            context.request_id,
            turns,
            tool_call_count,
        )

        return LangChainAgentResult(
            answer=answer,
            turns=turns,
            tool_call_count=tool_call_count,
        )

    @staticmethod
    def _load_create_agent():
        """延迟加载 LangChain v1 create_agent，避免污染 Native 导入路径。"""

        try:
            from langchain.agents import create_agent
        except ImportError as exc:
            raise RuntimeError(
                "LangChain Agent Runner requires langchain; "
                "install project requirements before using v2.1 framework integration"
            ) from exc

        return create_agent

    @staticmethod
    def _extract_messages(state: Mapping[str, Any]) -> list[Any]:
        """从 create_agent 输出状态中提取消息列表。"""

        messages = state.get("messages")
        if not isinstance(messages, (list, tuple)) or not messages:
            raise LangChainAgentError(
                "langchain agent returned invalid messages state"
            )
        return list(messages)

    @classmethod
    def _extract_final_answer(cls, messages: Sequence[Any]) -> str:
        """只提取最终 AI 文本，不读取 reasoning / hidden content blocks。"""

        for message in reversed(messages):
            if not cls._is_ai_message(message):
                continue
            if cls._extract_tool_calls(message):
                continue

            text = cls._extract_text_content(message).strip()
            if text:
                return text

        raise LangChainAgentError(
            "langchain agent completed without final answer"
        )

    @staticmethod
    def _is_ai_message(message: Any) -> bool:
        if isinstance(message, Mapping):
            return message.get("role") in {"assistant", "ai"}

        message_type = getattr(message, "type", None)
        return message_type in {"ai", "assistant"}

    @staticmethod
    def _extract_tool_calls(message: Any) -> list[Any]:
        if isinstance(message, Mapping):
            value = message.get("tool_calls")
        else:
            value = getattr(message, "tool_calls", None)

        if isinstance(value, (list, tuple)):
            return list(value)
        return []

    @staticmethod
    def _extract_text_content(message: Any) -> str:
        if isinstance(message, Mapping):
            content = message.get("content")
        else:
            content = getattr(message, "content", None)

        if isinstance(content, str):
            return content

        if not isinstance(content, (list, tuple)):
            return ""

        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
                continue

            if isinstance(block, Mapping):
                block_type = block.get("type")
                text = block.get("text")
                if block_type == "text" and isinstance(text, str):
                    parts.append(text)
                continue

            block_type = getattr(block, "type", None)
            text = getattr(block, "text", None)
            if block_type == "text" and isinstance(text, str):
                parts.append(text)

        return "".join(parts)

    @staticmethod
    def _normalize_message(message: str) -> str:
        """校验并标准化用户输入，与 Native Agent 保持相同边界。"""

        if not message or not message.strip():
            raise ValueError("message cannot be empty")
        return message.strip()
