import json
import logging
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from app.agent.context import ToolExecutionContext
from app.agent.model_response import (
    LLMToolCall,
    LLMToolExchange,
    LLMToolResponse,
    LLMToolResult,
)
from app.agent.run_event import (
    AgentMessageEvent,
    AgentRunEvent,
    AgentStatusEvent,
    AgentToolCallEvent,
    AgentToolResultEvent,
)
from app.agent.tool_dispatcher import ToolDispatcher
from app.agent.tools.base import (
    BaseAgentTool,
    ToolContract,
    ToolError,
)


logger = logging.getLogger(__name__)


class ToolCallingLLM(Protocol):
    """Native Agent Runner 依赖的最小模型能力接口。"""

    def chat_with_tool_history(
        self,
        *,
        message: str,
        tool_contracts: Sequence[ToolContract],
        history: Sequence[LLMToolExchange],
    ) -> LLMToolResponse:
        """根据 Tool 历史继续模型调用。"""
        ...


class AgentLoopError(RuntimeError):
    """Native Agent Loop 可识别的基础异常。"""

    code = "agent_loop_error"


class AgentTurnLimitError(AgentLoopError):
    """Agent 超过允许的最大模型轮次。"""

    code = "max_turns_exceeded"


class AgentToolCallLimitError(AgentLoopError):
    """Agent 超过允许的最大 Tool Call 数量。"""

    code = "max_tool_calls_exceeded"


class AgentRepeatedToolCallError(AgentLoopError):
    """Agent 在同一 Run 中重复请求完全相同的 Tool Call。"""

    code = "repeated_tool_call"


class AgentTimeoutError(AgentLoopError):
    """Agent 已超过本次 Run 的执行时限。"""

    code = "agent_timeout"


class NativeAgentResult(BaseModel):
    """一次最小 Native Agent Run 的稳定结果。"""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    answer: str = Field(min_length=1)
    turns: int = Field(ge=1)
    tool_call_count: int = Field(ge=0)


@dataclass(frozen=True)
class _ToolExecutionOutcome:
    """Runtime 内部一次 Tool 执行结果及可安全暴露的状态摘要。"""

    result: LLMToolResult
    ok: bool
    error_code: str | None = None


class NativeAgentRunner:
    """
    v2.0-B Native Agent Runtime。

    负责：
    Model -> Tool Call -> Dispatch -> Tool Result -> Model -> Final Answer。

    B5 起同时提供 provider-neutral 运行事件，供 SSE / WebSocket 等
    外层传输协议消费；Runtime 本身不生成 HTTP/SSE 文本。

    当前不负责：
    AgentRun 持久化、Checkpoint、重试策略、审批或框架集成。
    """

    def __init__(
        self,
        *,
        llm_service: ToolCallingLLM,
        tools: Sequence[BaseAgentTool[Any, Any]],
        max_turns: int = 4,
        max_tool_calls: int = 8,
        max_duration_seconds: float = 60.0,
    ) -> None:
        if max_turns <= 0:
            raise ValueError("max_turns must be greater than 0")
        if max_tool_calls <= 0:
            raise ValueError("max_tool_calls must be greater than 0")
        if max_duration_seconds <= 0:
            raise ValueError("max_duration_seconds must be greater than 0")

        self.llm_service = llm_service
        self.tools = list(tools)
        self.dispatcher = ToolDispatcher(self.tools)
        self.tool_contracts = [tool.get_contract() for tool in self.tools]
        self.max_turns = max_turns
        self.max_tool_calls = max_tool_calls
        self.max_duration_seconds = max_duration_seconds

    def run(
        self,
        *,
        db: Session,
        context: ToolExecutionContext,
        message: str,
    ) -> NativeAgentResult:
        """执行一次同步 Native Agent Run，并只返回最终结果。"""

        final_result: NativeAgentResult | None = None

        for event in self.run_events(
            db=db,
            context=context,
            message=message,
        ):
            if isinstance(event, AgentMessageEvent):
                final_result = NativeAgentResult(
                    answer=event.content,
                    turns=event.turns,
                    tool_call_count=event.tool_call_count,
                )

        if final_result is None:
            raise RuntimeError("agent run completed without final answer")

        return final_result

    def run_events(
        self,
        *,
        db: Session,
        context: ToolExecutionContext,
        message: str,
    ) -> Iterator[AgentRunEvent]:
        """
        执行一次同步 Native Agent Run，并产出安全运行事件。

        事件只描述阶段、Tool 名称、成功状态和最终答案；
        不暴露隐藏推理、Tool 参数正文或 Tool Result 正文。
        """

        normalized_message = self._normalize_message(message)
        started_at = time.monotonic()
        history: list[LLMToolExchange] = []
        seen_tool_calls: set[str] = set()
        tool_call_count = 0

        logger.info(
            "Native agent run started: request_id=%s tool_count=%d "
            "max_turns=%d max_tool_calls=%d",
            context.request_id,
            len(self.tools),
            self.max_turns,
            self.max_tool_calls,
        )

        for turn in range(1, self.max_turns + 1):
            self._ensure_within_deadline(started_at)

            yield AgentStatusEvent(
                stage="model",
                turn=turn,
            )

            response = self.llm_service.chat_with_tool_history(
                message=normalized_message,
                tool_contracts=self.tool_contracts,
                history=history,
            )

            if not response.tool_calls:
                answer = (response.content or "").strip()
                if not answer:
                    raise RuntimeError("model returned empty final answer")

                logger.info(
                    "Native agent run completed: request_id=%s turns=%d "
                    "tool_call_count=%d",
                    context.request_id,
                    turn,
                    tool_call_count,
                )

                yield AgentMessageEvent(
                    content=answer,
                    turns=turn,
                    tool_call_count=tool_call_count,
                )
                return

            if tool_call_count + len(response.tool_calls) > self.max_tool_calls:
                raise AgentToolCallLimitError(
                    "agent exceeded max_tool_calls"
                )

            tool_results: list[LLMToolResult] = []

            for tool_call in response.tool_calls:
                self._ensure_within_deadline(started_at)
                signature = self._tool_call_signature(tool_call)

                if signature in seen_tool_calls:
                    raise AgentRepeatedToolCallError(
                        f"repeated tool call: {tool_call.name}"
                    )

                seen_tool_calls.add(signature)
                tool_call_count += 1

                yield AgentToolCallEvent(
                    turn=turn,
                    call_id=tool_call.id,
                    tool_name=tool_call.name,
                )

                outcome = self._execute_tool_call(
                    db=db,
                    context=context,
                    tool_call=tool_call,
                )
                tool_results.append(outcome.result)

                yield AgentToolResultEvent(
                    turn=turn,
                    call_id=tool_call.id,
                    tool_name=tool_call.name,
                    ok=outcome.ok,
                    error_code=outcome.error_code,
                )

            history.append(
                LLMToolExchange(
                    response=response,
                    tool_results=tool_results,
                )
            )

        raise AgentTurnLimitError("agent exceeded max_turns")

    def _execute_tool_call(
        self,
        *,
        db: Session,
        context: ToolExecutionContext,
        tool_call: LLMToolCall,
    ) -> _ToolExecutionOutcome:
        """
        执行一次 Tool Call，并把 ToolError 转成可回填模型的安全结果。
        """

        error_code: str | None = None

        try:
            dispatch_result = self.dispatcher.dispatch(
                db=db,
                context=context,
                tool_call=tool_call,
            )
            payload: dict[str, Any] = {
                "ok": True,
                "result": dispatch_result.output,
            }
            ok = True

        except ToolError as exc:
            logger.warning(
                "Native agent tool call failed: request_id=%s "
                "tool_name=%s call_id=%s error_code=%s",
                context.request_id,
                tool_call.name,
                tool_call.id,
                exc.code,
            )
            error_code = exc.code
            ok = False
            payload = {
                "ok": False,
                "error": {
                    "code": exc.code,
                    "message": str(exc),
                },
            }

        result = LLMToolResult(
            call_id=tool_call.id,
            tool_name=tool_call.name,
            content_json=json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        )

        return _ToolExecutionOutcome(
            result=result,
            ok=ok,
            error_code=error_code,
        )

    def _ensure_within_deadline(self, started_at: float) -> None:
        """
        在进入下一次模型/Tool 调用前检查 Run 时限。

        当前同步实现无法中断已经阻塞中的 provider 调用，
        因此这是 operation-boundary deadline，而不是硬取消。
        """

        if time.monotonic() - started_at >= self.max_duration_seconds:
            raise AgentTimeoutError("agent exceeded max_duration_seconds")

    @staticmethod
    def _tool_call_signature(tool_call: LLMToolCall) -> str:
        """生成稳定签名，阻止同一 Run 重复执行完全相同的调用。"""

        try:
            arguments = json.loads(tool_call.arguments_json)
        except json.JSONDecodeError:
            canonical_arguments = tool_call.arguments_json
        else:
            canonical_arguments = json.dumps(
                arguments,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )

        return f"{tool_call.name}:{canonical_arguments}"

    @staticmethod
    def _normalize_message(message: str) -> str:
        """校验并标准化用户消息。"""

        if not message or not message.strip():
            raise ValueError("message cannot be empty")

        return message.strip()
