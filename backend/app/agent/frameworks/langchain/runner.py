"""LangChain v1 Single-Agent Candidate Runner。"""

import json
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from app.agent.agent_prompt import build_agent_tool_calling_system_prompt
from app.agent.context import ToolExecutionContext
from app.agent.frameworks.langchain.execution_observer import (
    LangChainToolExecutionObserver,
)
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
    """LangChain Graph 达到 framework recursion limit。"""

    code = "langchain_recursion_limit"


class LangChainAgentTurnLimitError(LangChainAgentError):
    """LangChain Candidate 超过允许的最大模型轮次。"""

    code = "max_turns_exceeded"


class LangChainAgentToolCallLimitError(LangChainAgentError):
    """LangChain Candidate 超过允许的最大 Tool Call 数量。"""

    code = "max_tool_calls_exceeded"


class LangChainAgentRepeatedToolCallError(LangChainAgentError):
    """LangChain Candidate 重复请求完全相同的 Tool Call。"""

    code = "repeated_tool_call"


class LangChainAgentTimeoutError(LangChainAgentError):
    """LangChain Candidate 超过本次 Run 的 operation-boundary 时限。"""

    code = "agent_timeout"


@dataclass
class _LangChainRuntimeBudget:
    """一次 LangChain Run 的请求级预算状态，不跨请求共享。"""

    max_model_turns: int
    max_tool_calls: int
    max_duration_seconds: float
    started_at: float | None = None
    model_turn_count: int = 0
    tool_call_count: int = 0
    seen_tool_calls: set[str] = field(default_factory=set)

    def start(self) -> None:
        """在 Graph 真正 invoke 前启动 deadline。"""

        self.started_at = time.monotonic()

    def ensure_within_deadline(self) -> None:
        """与 Native 一样，只在 operation boundary 检查总时限。"""

        if self.started_at is None:
            return
        if time.monotonic() - self.started_at >= self.max_duration_seconds:
            raise LangChainAgentTimeoutError(
                "langchain agent exceeded max_duration_seconds"
            )

    def register_model_turn(self) -> None:
        """在每次模型调用前登记业务模型轮次，与 Native max_turns 对齐。"""

        if self.model_turn_count >= self.max_model_turns:
            raise LangChainAgentTurnLimitError(
                "langchain agent exceeded max_model_turns"
            )
        self.model_turn_count += 1

    def register_tool_calls(self, tool_calls: Sequence[Any]) -> None:
        """在 Tool 执行前登记模型请求，并执行预算/重复调用保护。"""

        if not tool_calls:
            return

        if self.tool_call_count + len(tool_calls) > self.max_tool_calls:
            raise LangChainAgentToolCallLimitError(
                "langchain agent exceeded max_tool_calls"
            )

        signatures: list[str] = []
        for raw_call in tool_calls:
            signature = LangChainSingleAgentRunner._tool_call_signature(raw_call)
            if signature is not None and signature in self.seen_tool_calls:
                name = LangChainSingleAgentRunner._tool_call_name(raw_call)
                raise LangChainAgentRepeatedToolCallError(
                    f"repeated tool call: {name or 'unknown'}"
                )
            if signature is not None:
                signatures.append(signature)

        self.tool_call_count += len(tool_calls)
        self.seen_tool_calls.update(signatures)


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

    A4 起补齐与 Native 相同的 Tool / deadline Runtime Guard。
    A4.1 进一步把业务 max_model_turns 与 LangGraph recursion_limit 分离：
    - max_model_turns 是 Agent 业务预算，与 Native max_turns 对齐；
    - recursion_limit 只是 LangGraph 内部 super-step 的最后保险丝。

    当前仍刻意不负责：
    - 替换生产 /agent/chat；
    - AgentRun/ToolCall 生命周期持久化；
    - SSE；
    - 生产级 checkpoint / resume。
    """

    RUNNER_VERSION = "1.3.0"
    MIN_FRAMEWORK_RECURSION_LIMIT = 32
    FRAMEWORK_STEPS_PER_MODEL_TURN_HEADROOM = 8
    AGENT_NAME = "knowledge_assistant_langchain_candidate"

    def __init__(
        self,
        *,
        model: Any,
        tools: Sequence[BaseAgentTool[Any, Any]],
        max_model_turns: int = 4,
        recursion_limit: int | None = None,
        max_tool_calls: int = 8,
        max_duration_seconds: float = 60.0,
        agent_factory: Callable[..., LangChainAgentGraph] | None = None,
    ) -> None:
        if not tools:
            raise ValueError("tools cannot be empty")
        if max_model_turns <= 0:
            raise ValueError("max_model_turns must be greater than 0")
        if recursion_limit is not None and recursion_limit <= 0:
            raise ValueError("recursion_limit must be greater than 0")
        if max_tool_calls <= 0:
            raise ValueError("max_tool_calls must be greater than 0")
        if max_duration_seconds <= 0:
            raise ValueError("max_duration_seconds must be greater than 0")

        self.model = model
        self.tools = list(tools)
        self.tool_contracts = [tool.get_contract() for tool in self.tools]
        self.max_model_turns = max_model_turns
        self.recursion_limit = (
            recursion_limit
            if recursion_limit is not None
            else self._derive_framework_recursion_limit(max_model_turns)
        )
        self.max_tool_calls = max_tool_calls
        self.max_duration_seconds = max_duration_seconds
        self._tool_adapter = LangChainToolAdapter(self.tools)
        self._agent_factory = agent_factory

    def run(
        self,
        *,
        db: Session,
        context: ToolExecutionContext,
        message: str,
        observer: AgentRunObserver | None = None,
        execution_observer: LangChainToolExecutionObserver | None = None,
    ) -> LangChainAgentResult:
        """执行一次同步 LangChain Candidate Run。"""

        normalized_message = self._normalize_message(message)
        bound_tools = self._tool_adapter.bind_tools(
            db=db,
            context=context,
        )
        agent_factory = self._agent_factory or self._load_create_agent()

        runtime_budget = _LangChainRuntimeBudget(
            max_model_turns=self.max_model_turns,
            max_tool_calls=self.max_tool_calls,
            max_duration_seconds=self.max_duration_seconds,
        )
        middleware: list[Any] = [
            self._build_runtime_guard_middleware(runtime_budget)
        ]
        if observer is not None or execution_observer is not None:
            # Guard 放在最外层：
            # - after_model 中 Eval Observer 会先看到模型请求，再由 Guard 拒绝；
            # - wrap_tool_call 中 Guard 先放行后，execution observer 才记录实际执行。
            middleware.append(
                LangChainRunObserverBridge(
                    observer,
                    execution_observer=execution_observer,
                ).build_middleware()
            )

        agent_kwargs: dict[str, Any] = {
            "model": self.model,
            "tools": bound_tools,
            "system_prompt": build_agent_tool_calling_system_prompt(),
            "name": self.AGENT_NAME,
            "middleware": middleware,
        }

        graph = agent_factory(**agent_kwargs)

        logger.info(
            "LangChain agent run started: request_id=%s tool_count=%d "
            "max_model_turns=%d recursion_limit=%d max_tool_calls=%d "
            "max_duration_seconds=%.3f",
            context.request_id,
            len(bound_tools),
            self.max_model_turns,
            self.recursion_limit,
            self.max_tool_calls,
            self.max_duration_seconds,
        )

        runtime_budget.start()
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

    def _build_runtime_guard_middleware(
        self,
        runtime_budget: _LangChainRuntimeBudget,
    ) -> Any:
        """构建请求级 Guard Middleware，不把预算状态放到缓存 Runner 上。"""

        AgentMiddleware = self._load_agent_middleware()
        runner = self

        class RuntimeGuardMiddleware(AgentMiddleware):
            """在模型/Tool operation boundary 执行 Native 等价保护。"""

            def before_model(self, state, runtime):  # noqa: ANN001
                runtime_budget.ensure_within_deadline()
                runtime_budget.register_model_turn()
                return None

            def after_model(self, state, runtime):  # noqa: ANN001
                messages = state.get("messages") if isinstance(state, Mapping) else None
                if not isinstance(messages, (list, tuple)) or not messages:
                    return None

                tool_calls = runner._extract_tool_calls(messages[-1])
                runtime_budget.register_tool_calls(tool_calls)
                return None

            def wrap_tool_call(self, request, handler):  # noqa: ANN001
                runtime_budget.ensure_within_deadline()
                return handler(request)

        return RuntimeGuardMiddleware()

    @classmethod
    def _derive_framework_recursion_limit(cls, max_model_turns: int) -> int:
        """
        为 LangGraph super-step 预留宽松保险丝，不再把它当业务 turn budget。

        create_agent 的内部 graph / middleware 会让一次业务模型轮次消耗多个
        super-step，因此这里只按模型轮次提供显著 headroom；真正约束 Agent
        行为的是 max_model_turns / max_tool_calls / timeout。
        """

        return max(
            cls.MIN_FRAMEWORK_RECURSION_LIMIT,
            max_model_turns * cls.FRAMEWORK_STEPS_PER_MODEL_TURN_HEADROOM,
        )

    @staticmethod
    def _load_agent_middleware():
        """延迟加载 AgentMiddleware，保持 Native 导入路径不依赖 LangChain。"""

        try:
            from langchain.agents.middleware import AgentMiddleware
        except ImportError as exc:
            raise RuntimeError(
                "LangChain Runtime Guard requires langchain; "
                "install project requirements before using v2.1 framework integration"
            ) from exc
        return AgentMiddleware

    @classmethod
    def _tool_call_signature(cls, raw_call: Any) -> str | None:
        """生成与 Native 等价的 name + canonical arguments 重复调用签名。"""

        name = cls._tool_call_name(raw_call)
        if not name:
            return None

        if isinstance(raw_call, Mapping):
            arguments = raw_call.get("args", raw_call.get("arguments", {}))
        else:
            arguments = getattr(
                raw_call,
                "args",
                getattr(raw_call, "arguments", {}),
            )

        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError:
                canonical_arguments = arguments
            else:
                canonical_arguments = json.dumps(
                    parsed,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
        else:
            try:
                canonical_arguments = json.dumps(
                    arguments if arguments is not None else {},
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            except (TypeError, ValueError):
                canonical_arguments = str(arguments)

        return f"{name}:{canonical_arguments}"

    @staticmethod
    def _tool_call_name(raw_call: Any) -> str | None:
        if isinstance(raw_call, Mapping):
            value = raw_call.get("name")
        else:
            value = getattr(raw_call, "name", None)

        if not isinstance(value, str):
            return None
        normalized = value.strip()
        return normalized or None

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
