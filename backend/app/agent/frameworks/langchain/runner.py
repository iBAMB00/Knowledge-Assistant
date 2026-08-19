"""LangChain v1 Single-Agent Candidate Runner。"""

import json
import logging
import time
from collections.abc import Iterator, Mapping, Sequence
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
from app.agent.run_event import (
    AgentMessageEvent,
    AgentRunEvent,
    AgentStatusEvent,
    AgentToolCallEvent,
    AgentToolResultEvent,
)
from app.agent.run_observer import AgentRunObserver
from app.agent.tools.base import BaseAgentTool


logger = logging.getLogger(__name__)


class LangChainAgentGraph(Protocol):
    """Candidate Runner 真正依赖的最小 LangChain Graph 接口。"""

    def invoke(
        self,
        input: dict[str, Any],
        config: dict[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        """执行一次同步 Agent Graph。"""
        ...

    def stream(
        self,
        input: dict[str, Any],
        config: dict[str, Any] | None = None,
        *,
        stream_mode: str = "updates",
    ) -> Iterator[Mapping[str, Any]]:
        """按 Agent step 流式返回 Graph 状态更新。"""
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
    - 成为生产 /agent/chat 的默认 Runtime；
    - HTTP/SSE 文本编码；
    - 生产级 checkpoint / resume。

    A8 起额外提供 provider-neutral AgentRunEvent 流。只消费 LangGraph
    ``updates`` step stream，不消费 ``messages`` token stream，避免把 provider
    reasoning/thinking 内容带入对外事件。
    """

    RUNNER_VERSION = "1.4.0"
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
        graph, runtime_budget, bound_tool_count = self._build_graph(
            db=db,
            context=context,
            observer=observer,
            execution_observer=execution_observer,
        )

        logger.info(
            "LangChain agent run started: request_id=%s tool_count=%d "
            "max_model_turns=%d recursion_limit=%d max_tool_calls=%d "
            "max_duration_seconds=%.3f",
            context.request_id,
            bound_tool_count,
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

    def run_events(
        self,
        *,
        db: Session,
        context: ToolExecutionContext,
        message: str,
        observer: AgentRunObserver | None = None,
        execution_observer: LangChainToolExecutionObserver | None = None,
    ) -> Iterator[AgentRunEvent]:
        """执行 Candidate，并映射为与 Native 共用的安全运行事件。

        只使用 LangGraph ``updates``：
        - 首次模型调用前主动发 ``status``；
        - model update 中只读取结构化 Tool Call 或最终 text block；
        - tool update 中只读取 ToolMessage 的安全成功/错误元数据；
        - 不消费 token/reasoning stream，不输出 Tool 参数或 Tool Result 正文。
        """

        normalized_message = self._normalize_message(message)
        graph, runtime_budget, bound_tool_count = self._build_graph(
            db=db,
            context=context,
            observer=observer,
            execution_observer=execution_observer,
        )

        logger.info(
            "LangChain agent event stream started: request_id=%s tool_count=%d "
            "max_model_turns=%d recursion_limit=%d max_tool_calls=%d "
            "max_duration_seconds=%.3f",
            context.request_id,
            bound_tool_count,
            self.max_model_turns,
            self.recursion_limit,
            self.max_tool_calls,
            self.max_duration_seconds,
        )

        graph_stream: Iterator[Mapping[str, Any]] | None = None
        current_turn = 1
        tool_call_count = 0
        completed = False
        emitted_tool_calls: set[str] = set()
        emitted_tool_results: set[str] = set()
        pending_tool_calls: set[str] = set()
        tool_names_by_call_id: dict[str, str] = {}

        runtime_budget.start()
        yield AgentStatusEvent(stage="model", turn=current_turn)

        try:
            graph_stream = graph.stream(
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
                stream_mode="updates",
            )

            for raw_chunk in graph_stream:
                update_data = self._extract_stream_update_data(raw_chunk)
                if update_data is None:
                    continue

                saw_tool_result = False
                for state_update in update_data.values():
                    messages = self._extract_update_messages(state_update)
                    for item in messages:
                        if self._is_ai_message(item):
                            tool_calls = self._extract_tool_calls(item)
                            if tool_calls:
                                for raw_call in tool_calls:
                                    call_id = self._tool_call_id(raw_call)
                                    tool_name = self._tool_call_name(raw_call)
                                    if not call_id or not tool_name:
                                        raise LangChainAgentError(
                                            "langchain stream tool call missing id or name"
                                        )
                                    if call_id in emitted_tool_calls:
                                        continue

                                    emitted_tool_calls.add(call_id)
                                    pending_tool_calls.add(call_id)
                                    tool_names_by_call_id[call_id] = tool_name
                                    tool_call_count += 1
                                    yield AgentToolCallEvent(
                                        turn=current_turn,
                                        call_id=call_id,
                                        tool_name=tool_name,
                                    )
                                continue

                            if completed:
                                continue

                            answer = self._extract_text_content(item).strip()
                            if not answer:
                                continue

                            if observer is not None:
                                observer.on_final_answer(answer)

                            completed = True
                            yield AgentMessageEvent(
                                content=answer,
                                turns=current_turn,
                                tool_call_count=tool_call_count,
                            )
                            continue

                        if not self._is_tool_message(item):
                            continue

                        call_id = self._tool_message_call_id(item)
                        if not call_id or call_id in emitted_tool_results:
                            continue

                        tool_name = (
                            self._tool_message_name(item)
                            or tool_names_by_call_id.get(call_id)
                        )
                        if not tool_name:
                            raise LangChainAgentError(
                                "langchain stream tool result missing tool name"
                            )

                        ok, error_code = self._parse_stream_tool_result(item)
                        emitted_tool_results.add(call_id)
                        pending_tool_calls.discard(call_id)
                        saw_tool_result = True
                        yield AgentToolResultEvent(
                            turn=current_turn,
                            call_id=call_id,
                            tool_name=tool_name,
                            ok=ok,
                            duration_ms=0,
                            error_code=error_code,
                        )

                if (
                    saw_tool_result
                    and not pending_tool_calls
                    and not completed
                ):
                    current_turn += 1
                    yield AgentStatusEvent(
                        stage="model",
                        turn=current_turn,
                    )

            if not completed:
                raise LangChainAgentError(
                    "langchain agent event stream completed without final answer"
                )

            logger.info(
                "LangChain agent event stream completed: request_id=%s turns=%d "
                "tool_call_count=%d",
                context.request_id,
                current_turn,
                tool_call_count,
            )

        except RecursionError as exc:
            raise LangChainAgentLimitError(
                "langchain agent exceeded recursion_limit"
            ) from exc

        finally:
            self._close_iterator(graph_stream)

    def _build_graph(
        self,
        *,
        db: Session,
        context: ToolExecutionContext,
        observer: AgentRunObserver | None,
        execution_observer: LangChainToolExecutionObserver | None,
    ) -> tuple[LangChainAgentGraph, _LangChainRuntimeBudget, int]:
        """构建一次请求级 Graph 与 Runtime Budget，供 invoke/stream 共用。"""

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
            middleware.append(
                LangChainRunObserverBridge(
                    observer,
                    execution_observer=execution_observer,
                ).build_middleware()
            )

        graph = agent_factory(
            model=self.model,
            tools=bound_tools,
            system_prompt=build_agent_tool_calling_system_prompt(),
            name=self.AGENT_NAME,
            middleware=middleware,
        )
        return graph, runtime_budget, len(bound_tools)

    @staticmethod
    def _extract_stream_update_data(
        raw_chunk: Any,
    ) -> Mapping[str, Any] | None:
        """兼容 LangGraph updates 的默认格式与 v2 typed wrapper。"""

        if not isinstance(raw_chunk, Mapping):
            return None

        if raw_chunk.get("type") == "updates":
            data = raw_chunk.get("data")
            return data if isinstance(data, Mapping) else None

        return raw_chunk

    @staticmethod
    def _extract_update_messages(state_update: Any) -> list[Any]:
        if not isinstance(state_update, Mapping):
            return []
        messages = state_update.get("messages")
        if isinstance(messages, (list, tuple)):
            return list(messages)
        return []

    @staticmethod
    def _is_tool_message(message: Any) -> bool:
        if isinstance(message, Mapping):
            return message.get("role") == "tool" or message.get("type") == "tool"
        return getattr(message, "type", None) == "tool"

    @staticmethod
    def _tool_call_id(raw_call: Any) -> str | None:
        if isinstance(raw_call, Mapping):
            value = raw_call.get("id")
        else:
            value = getattr(raw_call, "id", None)
        if not isinstance(value, str):
            return None
        normalized = value.strip()
        return normalized or None

    @staticmethod
    def _tool_message_call_id(message: Any) -> str | None:
        if isinstance(message, Mapping):
            value = message.get("tool_call_id") or message.get("call_id")
        else:
            value = getattr(
                message,
                "tool_call_id",
                getattr(message, "call_id", None),
            )
        if not isinstance(value, str):
            return None
        normalized = value.strip()
        return normalized or None

    @staticmethod
    def _tool_message_name(message: Any) -> str | None:
        if isinstance(message, Mapping):
            value = message.get("name")
        else:
            value = getattr(message, "name", None)
        if not isinstance(value, str):
            return None
        normalized = value.strip()
        return normalized or None

    @classmethod
    def _parse_stream_tool_result(
        cls,
        message: Any,
    ) -> tuple[bool, str | None]:
        """只从 ToolMessage 的模型可见 JSON 提取安全 ok/error_code。"""

        content = cls._extract_text_content(message).strip()
        if not content:
            return True, None
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            return True, None
        if not isinstance(payload, Mapping) or payload.get("ok") is not False:
            return True, None

        error = payload.get("error")
        if not isinstance(error, Mapping):
            return False, "tool_error"
        code = error.get("code")
        if not isinstance(code, str) or not code.strip():
            return False, "tool_error"
        return False, code.strip()

    @staticmethod
    def _close_iterator(iterator: Any | None) -> None:
        if iterator is None:
            return
        close_method = getattr(iterator, "close", None)
        if callable(close_method):
            close_method()

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
