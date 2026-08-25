"""v2.3-A9 LangGraph Stateful Runtime with durable resume/HITL/cancellation.

LangGraph only owns orchestration here. Tool execution, trusted context and
business services continue to use the existing Agent core.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Callable, Protocol, TypedDict

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from app.agent.checkpoint import (
    AgentCheckpointStateTransitionError,
    AgentCheckpointWriter,
    AgentExecutionCheckpointPayload,
    AgentRecoveryLoader,
    AgentResumeStateError,
)
from app.agent.context import ToolExecutionContext
from app.agent.context_engine import AgentContextItem
from app.agent.hitl import (
    AgentApprovalRequirement,
    AgentApprovalStateError,
    AgentHITLLoader,
    AgentInterruptPolicy,
    AgentInterruptRequired,
    NoAgentInterruptPolicy,
)
from app.agent.observability.component import AgentComponentTracer
from app.agent.observability.contracts import (
    AgentComponentResult,
    AgentGraphExecutionMode,
)
from app.agent.observability.model import AgentModelTracer
from app.agent.model_response import (
    LLMToolCall,
    LLMToolExchange,
    LLMToolResponse,
)
from app.agent.native_agent import (
    AgentRepeatedToolCallError,
    AgentToolCallLimitError,
    AgentTurnLimitError,
    NativeAgentRunner,
    ToolCallingLLM,
)
from app.agent.run_event import (
    AgentMessageEvent,
    AgentRunEvent,
    AgentStatusEvent,
    AgentToolCallEvent,
    AgentToolResultEvent,
)
from app.agent.run_observer import AgentRunObserver
from app.agent.run_control import (
    AgentCancellationProbe,
    AgentRunCancellationError,
)
from app.agent.state import AgentState
from app.agent.tools.base import BaseAgentTool
from app.constants.agent_state_status import AgentStateStatus
from app.constants.conversation_message_role import ConversationMessageRole
from app.schemas.conversation_contract import ConversationMessagePayload


class _CompiledGraph(Protocol):
    """A5 只依赖 compiled graph 的 invoke / stream 稳定表面。"""

    def invoke(
        self,
        input: Mapping[str, Any],
        config: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        ...

    def stream(
        self,
        input: Mapping[str, Any],
        config: Mapping[str, Any] | None = None,
        *,
        stream_mode: str,
    ) -> Iterator[Mapping[str, Any]]:
        ...


class _StateGraphBuilder(Protocol):
    """A5 Runner 真正依赖的最小 StateGraph Builder Contract。"""

    def add_node(self, node: str, action: Callable[..., Any]) -> Any:
        ...

    def add_edge(self, start_key: Any, end_key: Any) -> Any:
        ...

    def add_conditional_edges(
        self,
        source: Any,
        path: Callable[..., Any],
        path_map: Mapping[str, Any],
    ) -> Any:
        ...

    def compile(self) -> _CompiledGraph:
        ...


class _LangGraphExecutionState(TypedDict):
    """
    LangGraph 的可恢复 Graph Execution State。

    ``agent_state`` 是 A1 冻结的框架无关状态；其余字段是 Tool Loop 续跑
    所需的 provider-neutral 编排状态。A6 已把这些字段纳入 durable
    checkpoint；A7 从同一 Contract 恢复，不保存隐藏推理过程。
    """

    agent_state: AgentState
    history: tuple[LLMToolExchange, ...]
    pending_tool_calls: tuple[LLMToolCall, ...]
    last_model_response: LLMToolResponse | None
    tool_observations: tuple[AgentToolResultEvent, ...]
    final_answer: str | None
    turn: int
    tool_call_count: int
    seen_tool_call_signatures: tuple[str, ...]
    pending_approvals: tuple[AgentApprovalRequirement, ...]
    approved_call_ids: tuple[str, ...]
    rejected_call_ids: tuple[str, ...]


class LangGraphStatefulResult(BaseModel):
    """Minimal Stateful Runtime 的同步结果。"""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    answer: str = Field(min_length=1)
    turns: int = Field(ge=1)
    tool_call_count: int = Field(ge=0)
    state: AgentState


class LangGraphStatefulRunner(NativeAgentRunner):
    """
    v2.3-A9 显式 StateGraph + durable checkpoint + resume + HITL + Cancel Candidate。

    只替换 Agent Loop 的编排方式：

        START -> Agent Node -> needs tools?
                          | yes -> Approval Node -> approved?
                          |                         | yes -> Tool Node -> Agent Node
                          |                         | no  -> WAITING -> END
                          | no  -> END

    Tool Contract、ToolDispatcher、Trusted Context、安全 ToolError 回填、
    max_turns / max_tool_calls / timeout / repeated-call protection 全部沿用
    Native Baseline，避免为了“上 LangGraph”重写既有业务边界。

    A6 通过框架无关 checkpoint writer 在关键 Node 边界落库；A7 新增
    recovery loader；A8 再加入 Approval Node 与 WAITING checkpoint。审批策略
    仍通过 framework-neutral Protocol 注入，v2.6 再负责风险治理与审批矩阵。
    """

    RUNNER_VERSION = "0.4.0"
    GRAPH_VERSION = "1.3"

    AGENT_NODE = "agent"
    APPROVAL_NODE = "approval"
    TOOL_NODE = "tools"

    def __init__(
        self,
        *,
        llm_service: ToolCallingLLM,
        tools: Sequence[BaseAgentTool[Any, Any]],
        max_turns: int = 4,
        max_tool_calls: int = 8,
        max_duration_seconds: float = 60.0,
        checkpoint_writer: AgentCheckpointWriter | None = None,
        recovery_loader: AgentRecoveryLoader | None = None,
        interrupt_policy: AgentInterruptPolicy | None = None,
        hitl_loader: AgentHITLLoader | None = None,
        cancellation_probe: AgentCancellationProbe | None = None,
    ) -> None:
        super().__init__(
            llm_service=llm_service,
            tools=tools,
            max_turns=max_turns,
            max_tool_calls=max_tool_calls,
            max_duration_seconds=max_duration_seconds,
        )
        self.checkpoint_writer = checkpoint_writer
        self.recovery_loader = recovery_loader
        self.interrupt_policy = interrupt_policy or NoAgentInterruptPolicy()
        self.hitl_loader = hitl_loader
        self.cancellation_probe = cancellation_probe
        self._tool_contract_by_name = {
            contract.name: contract for contract in self.tool_contracts
        }

    def run(
        self,
        *,
        db: Session,
        context: ToolExecutionContext,
        message: str,
        state: AgentState,
        observer: AgentRunObserver | None = None,
        supporting_context: Sequence[AgentContextItem] = (),
        model_tracer: AgentModelTracer | None = None,
        component_tracer: AgentComponentTracer | None = None,
    ) -> LangGraphStatefulResult:
        """执行一次 Minimal StateGraph，并返回最终框架无关 AgentState。"""

        graph, initial_state = self._build_graph_execution(
            db=db,
            context=context,
            message=message,
            state=state,
            observer=observer,
            supporting_context=supporting_context,
            model_tracer=model_tracer,
            component_tracer=component_tracer,
        )
        final_raw = graph.invoke(
            initial_state,
            config=self._graph_config(),
        )
        final_state = self._coerce_graph_state(final_raw)
        self._raise_if_interrupted(final_state)
        answer = (final_state["final_answer"] or "").strip()

        if not answer:
            raise RuntimeError("langgraph agent completed without final answer")

        return LangGraphStatefulResult(
            answer=answer,
            turns=final_state["turn"],
            tool_call_count=final_state["tool_call_count"],
            state=final_state["agent_state"],
        )

    def resume(
        self,
        *,
        db: Session,
        context: ToolExecutionContext,
        thread_id: str,
        observer: AgentRunObserver | None = None,
        supporting_context: Sequence[AgentContextItem] = (),
        model_tracer: AgentModelTracer | None = None,
        component_tracer: AgentComponentTracer | None = None,
    ) -> LangGraphStatefulResult:
        """从最新 durable checkpoint 继续一次中断的 RUNNING Thread。"""

        initial_state = self._prepare_resume_initial_state(
            db=db,
            context=context,
            thread_id=thread_id,
        )
        task = initial_state["agent_state"].task
        if not task:
            raise AgentResumeStateError(
                "resume checkpoint is missing task"
            )

        graph, resumed_state = self._build_graph_execution(
            db=db,
            context=context,
            message=task,
            state=initial_state["agent_state"],
            observer=observer,
            supporting_context=supporting_context,
            model_tracer=model_tracer,
            component_tracer=component_tracer,
            initial_state_override=initial_state,
        )
        final_raw = graph.invoke(
            resumed_state,
            config=self._graph_config(),
        )
        final_state = self._coerce_graph_state(final_raw)
        self._raise_if_interrupted(final_state)
        answer = (final_state["final_answer"] or "").strip()

        if not answer:
            raise RuntimeError(
                "langgraph resumed without final answer"
            )

        return LangGraphStatefulResult(
            answer=answer,
            turns=final_state["turn"],
            tool_call_count=final_state["tool_call_count"],
            state=final_state["agent_state"],
        )

    def resume_after_approval(
        self,
        *,
        db: Session,
        context: ToolExecutionContext,
        thread_id: str,
        observer: AgentRunObserver | None = None,
        supporting_context: Sequence[AgentContextItem] = (),
        model_tracer: AgentModelTracer | None = None,
        component_tracer: AgentComponentTracer | None = None,
    ) -> LangGraphStatefulResult:
        """从已批准的 WAITING checkpoint 继续执行 pending ToolCall。"""

        initial_state = self._prepare_approved_initial_state(
            db=db,
            context=context,
            thread_id=thread_id,
        )
        task = initial_state["agent_state"].task
        if not task:
            raise AgentApprovalStateError(
                "approved checkpoint is missing task"
            )

        graph, resumed_state = self._build_graph_execution(
            db=db,
            context=context,
            message=task,
            state=initial_state["agent_state"],
            observer=observer,
            supporting_context=supporting_context,
            model_tracer=model_tracer,
            component_tracer=component_tracer,
            initial_state_override=initial_state,
        )
        final_raw = graph.invoke(
            resumed_state,
            config=self._graph_config(),
        )
        final_state = self._coerce_graph_state(final_raw)
        self._raise_if_interrupted(final_state)
        answer = (final_state["final_answer"] or "").strip()

        if not answer:
            raise RuntimeError(
                "langgraph approval resume completed without final answer"
            )

        return LangGraphStatefulResult(
            answer=answer,
            turns=final_state["turn"],
            tool_call_count=final_state["tool_call_count"],
            state=final_state["agent_state"],
        )

    def resume_after_approval_events(
        self,
        *,
        db: Session,
        context: ToolExecutionContext,
        thread_id: str,
        observer: AgentRunObserver | None = None,
        supporting_context: Sequence[AgentContextItem] = (),
        model_tracer: AgentModelTracer | None = None,
        component_tracer: AgentComponentTracer | None = None,
    ) -> Iterator[AgentRunEvent]:
        """从已批准 WAITING checkpoint 续跑，并输出既有安全事件。"""

        initial_state = self._prepare_approved_initial_state(
            db=db,
            context=context,
            thread_id=thread_id,
        )
        task = initial_state["agent_state"].task
        if not task:
            raise AgentApprovalStateError(
                "approved checkpoint is missing task"
            )

        graph, resumed_state = self._build_graph_execution(
            db=db,
            context=context,
            message=task,
            state=initial_state["agent_state"],
            observer=observer,
            supporting_context=supporting_context,
            model_tracer=model_tracer,
            component_tracer=component_tracer,
            initial_state_override=initial_state,
        )

        # 新连接先重放安全 ToolCall 元数据，仍不暴露 arguments_json。
        for tool_call in resumed_state["pending_tool_calls"]:
            yield AgentToolCallEvent(
                turn=max(1, resumed_state["turn"]),
                call_id=tool_call.id,
                tool_name=tool_call.name,
            )

        graph_stream: Iterator[Mapping[str, Any]] | None = None
        completed = False
        interrupted_approvals: tuple[AgentApprovalRequirement, ...] = ()

        try:
            graph_stream = graph.stream(
                resumed_state,
                config=self._graph_config(),
                stream_mode="updates",
            )

            for raw_update in graph_stream:
                for node_name, patch in raw_update.items():
                    if not isinstance(patch, Mapping):
                        continue

                    if node_name == self.AGENT_NODE:
                        turn = int(patch.get("turn", 0))
                        if turn <= 0:
                            continue
                        yield AgentStatusEvent(stage="model", turn=turn)

                        for tool_call in patch.get(
                            "pending_tool_calls",
                            (),
                        ):
                            if isinstance(tool_call, LLMToolCall):
                                yield AgentToolCallEvent(
                                    turn=turn,
                                    call_id=tool_call.id,
                                    tool_name=tool_call.name,
                                )

                        answer = patch.get("final_answer")
                        if isinstance(answer, str) and answer.strip():
                            completed = True
                            yield AgentMessageEvent(
                                content=answer.strip(),
                                turns=turn,
                                tool_call_count=int(
                                    patch.get("tool_call_count", 0)
                                ),
                            )

                    elif node_name == self.APPROVAL_NODE:
                        state_value = patch.get("agent_state")
                        if (
                            isinstance(state_value, AgentState)
                            and state_value.status is AgentStateStatus.WAITING
                        ):
                            interrupted_approvals = tuple(
                                item
                                for item in patch.get(
                                    "pending_approvals",
                                    (),
                                )
                                if isinstance(
                                    item,
                                    AgentApprovalRequirement,
                                )
                            )

                    elif node_name == self.TOOL_NODE:
                        for observation in patch.get(
                            "tool_observations",
                            (),
                        ):
                            if isinstance(
                                observation,
                                AgentToolResultEvent,
                            ):
                                yield observation

            if interrupted_approvals:
                raise AgentInterruptRequired(
                    thread_id=resumed_state["agent_state"].thread.thread_id,
                    approvals=interrupted_approvals,
                )
            if not completed:
                raise RuntimeError(
                    "langgraph approval resume event stream completed "
                    "without final answer"
                )
        finally:
            self._close_iterator(graph_stream)

    def resume_events(
        self,
        *,
        db: Session,
        context: ToolExecutionContext,
        thread_id: str,
        observer: AgentRunObserver | None = None,
        supporting_context: Sequence[AgentContextItem] = (),
        model_tracer: AgentModelTracer | None = None,
        component_tracer: AgentComponentTracer | None = None,
    ) -> Iterator[AgentRunEvent]:
        """从最新 checkpoint 恢复，并继续输出既有安全 SSE Event。"""

        initial_state = self._prepare_resume_initial_state(
            db=db,
            context=context,
            thread_id=thread_id,
        )
        task = initial_state["agent_state"].task
        if not task:
            raise AgentResumeStateError(
                "resume checkpoint is missing task"
            )

        graph, resumed_state = self._build_graph_execution(
            db=db,
            context=context,
            message=task,
            state=initial_state["agent_state"],
            observer=observer,
            supporting_context=supporting_context,
            model_tracer=model_tracer,
            component_tracer=component_tracer,
            initial_state_override=initial_state,
        )

        # 如果上次崩在“模型已决定 Tool、但 Tool 结果尚未持久化”的边界，
        # 新 SSE 连接先重放安全的 ToolCall 元数据，让后续 ToolResult 有上下文。
        for tool_call in resumed_state["pending_tool_calls"]:
            yield AgentToolCallEvent(
                turn=max(1, resumed_state["turn"]),
                call_id=tool_call.id,
                tool_name=tool_call.name,
            )

        graph_stream: Iterator[Mapping[str, Any]] | None = None
        completed = False
        interrupted_approvals: tuple[AgentApprovalRequirement, ...] = ()

        try:
            graph_stream = graph.stream(
                resumed_state,
                config=self._graph_config(),
                stream_mode="updates",
            )

            for raw_update in graph_stream:
                for node_name, patch in raw_update.items():
                    if not isinstance(patch, Mapping):
                        continue

                    if node_name == self.AGENT_NODE:
                        turn = int(patch.get("turn", 0))
                        if turn <= 0:
                            continue

                        yield AgentStatusEvent(stage="model", turn=turn)

                        for tool_call in patch.get(
                            "pending_tool_calls",
                            (),
                        ):
                            if isinstance(tool_call, LLMToolCall):
                                yield AgentToolCallEvent(
                                    turn=turn,
                                    call_id=tool_call.id,
                                    tool_name=tool_call.name,
                                )

                        answer = patch.get("final_answer")
                        if isinstance(answer, str) and answer.strip():
                            completed = True
                            yield AgentMessageEvent(
                                content=answer.strip(),
                                turns=turn,
                                tool_call_count=int(
                                    patch.get("tool_call_count", 0)
                                ),
                            )

                    elif node_name == self.APPROVAL_NODE:
                        state_value = patch.get("agent_state")
                        if (
                            isinstance(state_value, AgentState)
                            and state_value.status is AgentStateStatus.WAITING
                        ):
                            interrupted_approvals = tuple(
                                item
                                for item in patch.get(
                                    "pending_approvals",
                                    (),
                                )
                                if isinstance(
                                    item,
                                    AgentApprovalRequirement,
                                )
                            )

                    elif node_name == self.TOOL_NODE:
                        for observation in patch.get(
                            "tool_observations",
                            (),
                        ):
                            if isinstance(
                                observation,
                                AgentToolResultEvent,
                            ):
                                yield observation

            if interrupted_approvals:
                raise AgentInterruptRequired(
                    thread_id=resumed_state["agent_state"].thread.thread_id,
                    approvals=interrupted_approvals,
                )
            if not completed:
                raise RuntimeError(
                    "langgraph resume event stream completed "
                    "without final answer"
                )

        finally:
            self._close_iterator(graph_stream)

    def run_events(
        self,
        *,
        db: Session,
        context: ToolExecutionContext,
        message: str,
        state: AgentState,
        observer: AgentRunObserver | None = None,
        supporting_context: Sequence[AgentContextItem] = (),
        model_tracer: AgentModelTracer | None = None,
        component_tracer: AgentComponentTracer | None = None,
    ) -> Iterator[AgentRunEvent]:
        """执行 Minimal StateGraph，并继续输出现有 provider-neutral 安全事件。"""

        graph, initial_state = self._build_graph_execution(
            db=db,
            context=context,
            message=message,
            state=state,
            observer=observer,
            supporting_context=supporting_context,
            model_tracer=model_tracer,
            component_tracer=component_tracer,
        )
        graph_stream: Iterator[Mapping[str, Any]] | None = None
        completed = False
        interrupted_approvals: tuple[AgentApprovalRequirement, ...] = ()

        try:
            graph_stream = graph.stream(
                initial_state,
                config=self._graph_config(),
                stream_mode="updates",
            )

            for raw_update in graph_stream:
                for node_name, patch in raw_update.items():
                    if not isinstance(patch, Mapping):
                        continue

                    if node_name == self.AGENT_NODE:
                        turn = int(patch.get("turn", 0))
                        if turn <= 0:
                            continue

                        yield AgentStatusEvent(stage="model", turn=turn)

                        for tool_call in patch.get(
                            "pending_tool_calls",
                            (),
                        ):
                            if isinstance(tool_call, LLMToolCall):
                                yield AgentToolCallEvent(
                                    turn=turn,
                                    call_id=tool_call.id,
                                    tool_name=tool_call.name,
                                )

                        answer = patch.get("final_answer")
                        if isinstance(answer, str) and answer.strip():
                            completed = True
                            yield AgentMessageEvent(
                                content=answer.strip(),
                                turns=turn,
                                tool_call_count=int(
                                    patch.get("tool_call_count", 0)
                                ),
                            )

                    elif node_name == self.APPROVAL_NODE:
                        state_value = patch.get("agent_state")
                        if (
                            isinstance(state_value, AgentState)
                            and state_value.status is AgentStateStatus.WAITING
                        ):
                            interrupted_approvals = tuple(
                                item
                                for item in patch.get(
                                    "pending_approvals",
                                    (),
                                )
                                if isinstance(
                                    item,
                                    AgentApprovalRequirement,
                                )
                            )

                    elif node_name == self.TOOL_NODE:
                        for observation in patch.get(
                            "tool_observations",
                            (),
                        ):
                            if isinstance(
                                observation,
                                AgentToolResultEvent,
                            ):
                                yield observation

            if interrupted_approvals:
                raise AgentInterruptRequired(
                    thread_id=initial_state["agent_state"].thread.thread_id,
                    approvals=interrupted_approvals,
                )
            if not completed:
                raise RuntimeError(
                    "langgraph agent event stream completed without final answer"
                )

        finally:
            self._close_iterator(graph_stream)

    def _build_graph_execution(
        self,
        *,
        db: Session,
        context: ToolExecutionContext,
        message: str,
        state: AgentState,
        observer: AgentRunObserver | None,
        supporting_context: Sequence[AgentContextItem] = (),
        model_tracer: AgentModelTracer | None = None,
        component_tracer: AgentComponentTracer | None = None,
        initial_state_override: _LangGraphExecutionState | None = None,
    ) -> tuple[_CompiledGraph, _LangGraphExecutionState]:
        normalized_message = self._normalize_message(message)

        if initial_state_override is None:
            initial_state = self._prepare_initial_state(
                context=context,
                message=normalized_message,
                state=state,
            )
        else:
            initial_state = self._coerce_graph_state(
                initial_state_override
            )
            resumed_task = initial_state["agent_state"].task
            if resumed_task != normalized_message:
                raise AgentResumeStateError(
                    "resume task does not match checkpoint task"
                )

        graph_execution_mode = (
            AgentGraphExecutionMode.FRESH
            if initial_state_override is None
            else AgentGraphExecutionMode.RESUME
        )

        # Fresh Run 与 Resume 都先写一个新的 AgentRun attempt 起始边界。
        # Fresh turn 只允许从终态开启；Resume 只允许接管 RUNNING/WAITING。
        # 这个显式边界同时用于 AgentRun fencing，避免旧 Runner 在取消后
        # 或下一轮已经开始后继续反写 checkpoint。
        previous_statuses = (
            {
                AgentStateStatus.SUCCEEDED,
                AgentStateStatus.FAILED,
                AgentStateStatus.CANCELLED,
            }
            if initial_state_override is None
            else {
                AgentStateStatus.RUNNING,
                AgentStateStatus.WAITING,
            }
        )
        self._save_checkpoint_if_enabled(
            db,
            initial_state,
            allowed_previous_statuses=previous_statuses,
            new_execution_attempt=True,
        )
        started_at = time.monotonic()

        def agent_node(
            graph_state: _LangGraphExecutionState,
        ) -> dict[str, Any]:
            self._ensure_within_deadline(started_at)
            self._ensure_not_cancelled(
                db=db,
                context=context,
                graph_state=graph_state,
            )

            turn = graph_state["turn"] + 1
            if turn > self.max_turns:
                raise AgentTurnLimitError("agent exceeded max_turns")

            response = self._chat_with_tool_history(
                message=(
                    graph_state["agent_state"].task
                    or normalized_message
                ),
                history=graph_state["history"],
                supporting_context=supporting_context,
                model_tracer=model_tracer,
                model_turn=turn,
            )
            # 外部 Cancel 可能发生在阻塞模型调用期间；返回后再次检查，
            # 防止旧内存状态覆盖 durable CANCELLED checkpoint。
            self._ensure_not_cancelled(
                db=db,
                context=context,
                graph_state=graph_state,
            )

            if response.tool_calls:
                if observer is not None:
                    for tool_call in response.tool_calls:
                        observer.on_tool_call_requested(tool_call)

                next_tool_count = (
                    graph_state["tool_call_count"]
                    + len(response.tool_calls)
                )
                if next_tool_count > self.max_tool_calls:
                    raise AgentToolCallLimitError(
                        "agent exceeded max_tool_calls"
                    )

                seen = set(
                    graph_state["seen_tool_call_signatures"]
                )
                signatures: list[str] = []

                for tool_call in response.tool_calls:
                    signature = self._tool_call_signature(tool_call)
                    if signature in seen:
                        raise AgentRepeatedToolCallError(
                            f"repeated tool call: {tool_call.name}"
                        )
                    seen.add(signature)
                    signatures.append(signature)

                patch = {
                    "agent_state": graph_state[
                        "agent_state"
                    ].model_copy(
                        update={"status": AgentStateStatus.RUNNING}
                    ),
                    "pending_tool_calls": tuple(
                        response.tool_calls
                    ),
                    "last_model_response": response,
                    "tool_observations": (),
                    "final_answer": None,
                    "turn": turn,
                    "tool_call_count": next_tool_count,
                    "seen_tool_call_signatures": (
                        *graph_state[
                            "seen_tool_call_signatures"
                        ],
                        *signatures,
                    ),
                }
                self._save_checkpoint_if_enabled(
                    db,
                    self._merge_graph_state(graph_state, patch),
                )
                return patch

            answer = (response.content or "").strip()
            if not answer:
                raise RuntimeError("model returned empty final answer")

            current_state = graph_state["agent_state"]
            final_agent_state = current_state.model_copy(
                update={
                    "status": AgentStateStatus.SUCCEEDED,
                    "messages": (
                        *current_state.messages,
                        ConversationMessagePayload(
                            role=ConversationMessageRole.ASSISTANT,
                            content=answer,
                        ),
                    ),
                    "last_error_code": None,
                }
            )

            if observer is not None:
                observer.on_final_answer(answer)

            patch = {
                "agent_state": final_agent_state,
                "pending_tool_calls": (),
                "last_model_response": response,
                "tool_observations": (),
                "final_answer": answer,
                "turn": turn,
                "tool_call_count": graph_state[
                    "tool_call_count"
                ],
            }
            self._save_checkpoint_if_enabled(
                db,
                self._merge_graph_state(graph_state, patch),
            )
            return patch

        def approval_node(
            graph_state: _LangGraphExecutionState,
        ) -> dict[str, Any]:
            self._ensure_not_cancelled(
                db=db,
                context=context,
                graph_state=graph_state,
            )
            pending_calls = graph_state["pending_tool_calls"]
            if not pending_calls:
                raise RuntimeError(
                    "approval node requires pending tool calls"
                )

            approved_ids = set(graph_state["approved_call_ids"])
            rejected_ids = set(graph_state["rejected_call_ids"])
            requirements: list[AgentApprovalRequirement] = []

            for tool_call in pending_calls:
                contract = self._tool_contract_by_name.get(
                    tool_call.name
                )
                if contract is None:
                    # ToolDispatcher 会在真正执行时返回稳定 tool_not_found；
                    # Approval Node 不抢占既有错误语义。
                    continue

                requirement = (
                    self.interrupt_policy.get_approval_requirement(
                        tool_call=tool_call,
                        tool_contract=contract,
                    )
                )
                if requirement is None:
                    continue
                requirements.append(requirement)

            requirement_ids = {item.call_id for item in requirements}
            if not requirement_ids:
                return {
                    "pending_approvals": (),
                    "approved_call_ids": (),
                    "rejected_call_ids": (),
                }

            if rejected_ids & requirement_ids:
                cancelled_state = graph_state[
                    "agent_state"
                ].model_copy(
                    update={
                        "status": AgentStateStatus.CANCELLED,
                        "last_error_code": "approval_rejected",
                    }
                )
                patch = {
                    "agent_state": cancelled_state,
                    "pending_approvals": tuple(requirements),
                }
                self._save_checkpoint_if_enabled(
                    db,
                    self._merge_graph_state(graph_state, patch),
                )
                return patch

            unresolved = requirement_ids - approved_ids
            if unresolved:
                waiting_state = graph_state[
                    "agent_state"
                ].model_copy(
                    update={
                        "status": AgentStateStatus.WAITING,
                        "last_error_code": None,
                    }
                )
                patch = {
                    "agent_state": waiting_state,
                    "pending_approvals": tuple(requirements),
                }
                self._save_checkpoint_if_enabled(
                    db,
                    self._merge_graph_state(graph_state, patch),
                )
                return patch

            running_state = graph_state[
                "agent_state"
            ].model_copy(
                update={
                    "status": AgentStateStatus.RUNNING,
                    "last_error_code": None,
                }
            )
            return {
                "agent_state": running_state,
                "pending_approvals": tuple(requirements),
            }

        def tool_node(
            graph_state: _LangGraphExecutionState,
        ) -> dict[str, Any]:
            self._ensure_not_cancelled(
                db=db,
                context=context,
                graph_state=graph_state,
            )
            response = graph_state["last_model_response"]
            pending_calls = graph_state["pending_tool_calls"]

            if response is None or not pending_calls:
                raise RuntimeError(
                    "tool node requires pending tool calls"
                )

            tool_results = []
            observations: list[AgentToolResultEvent] = []

            for tool_call in pending_calls:
                self._ensure_within_deadline(started_at)
                self._ensure_not_cancelled(
                    db=db,
                    context=context,
                    graph_state=graph_state,
                )
                tool_started_at = time.perf_counter()
                outcome = self._execute_tool_call(
                    db=db,
                    context=context,
                    tool_call=tool_call,
                    component_tracer=component_tracer,
                    turn=max(1, graph_state["turn"]),
                )
                # 无法强杀正在阻塞的外部 Tool；但 Tool 返回后立即再次检查，
                # 可阻止后续 Tool 和 stale checkpoint 继续推进。
                self._ensure_not_cancelled(
                    db=db,
                    context=context,
                    graph_state=graph_state,
                )
                duration_ms = max(
                    0,
                    int(
                        (time.perf_counter() - tool_started_at)
                        * 1000
                    ),
                )
                tool_results.append(outcome.result)

                if observer is not None:
                    observer.on_tool_result(
                        call_id=tool_call.id,
                        tool_name=tool_call.name,
                        ok=outcome.ok,
                        error_code=outcome.error_code,
                        evidence_refs=outcome.evidence_refs,
                    )

                observations.append(
                    AgentToolResultEvent(
                        turn=graph_state["turn"],
                        call_id=tool_call.id,
                        tool_name=tool_call.name,
                        ok=outcome.ok,
                        duration_ms=duration_ms,
                        error_code=outcome.error_code,
                    )
                )

            exchange = LLMToolExchange(
                response=response,
                tool_results=tool_results,
            )

            patch = {
                "history": (
                    *graph_state["history"],
                    exchange,
                ),
                "pending_tool_calls": (),
                "tool_observations": tuple(observations),
                "pending_approvals": (),
                "approved_call_ids": (),
                "rejected_call_ids": (),
            }
            self._save_checkpoint_if_enabled(
                db,
                self._merge_graph_state(graph_state, patch),
            )
            return patch

        def route_from_start(
            graph_state: _LangGraphExecutionState,
        ) -> str:
            # Fresh Run / Tool 已完成的 Resume 都进入 Agent。
            # pending ToolCall 一律先经过 Approval Node；默认 READ_ONLY 策略
            # 会直接放行，WAITING Resume 则在这里消费 durable approval。
            if graph_state["pending_tool_calls"]:
                return "approval"
            if graph_state["final_answer"]:
                return "end"
            return "agent"

        def route_after_agent(
            graph_state: _LangGraphExecutionState,
        ) -> str:
            if graph_state["pending_tool_calls"]:
                return "approval"
            return "end"

        def route_after_approval(
            graph_state: _LangGraphExecutionState,
        ) -> str:
            status = graph_state["agent_state"].status
            if status in (
                AgentStateStatus.WAITING,
                AgentStateStatus.CANCELLED,
            ):
                return "end"
            return "tools"

        state_graph_factory, start_symbol, end_symbol = (
            self._load_langgraph_components()
        )
        builder: _StateGraphBuilder = state_graph_factory(
            _LangGraphExecutionState
        )
        def traced_node(node_name: str, action: Callable[..., Any]):
            if component_tracer is None:
                return action

            def wrapped(graph_state: _LangGraphExecutionState) -> dict[str, Any]:
                raw_turn = int(graph_state.get("turn", 0))
                observed_turn = max(
                    1,
                    raw_turn + (1 if node_name == self.AGENT_NODE else 0),
                )
                handle = component_tracer.start_graph_node(
                    node_name=node_name,
                    execution_mode=graph_execution_mode,
                    turn=observed_turn,
                )
                try:
                    patch = action(graph_state)
                except Exception as exc:
                    handle.finish(
                        ok=False,
                        error_code=type(exc).__name__,
                    )
                    raise

                state_value = patch.get("agent_state")
                state_status = (
                    state_value.status.value
                    if isinstance(state_value, AgentState)
                    else None
                )
                if node_name == self.AGENT_NODE:
                    next_route = (
                        "approval"
                        if patch.get("pending_tool_calls")
                        else "end"
                    )
                elif node_name == self.APPROVAL_NODE:
                    next_route = (
                        "end"
                        if state_status in {
                            AgentStateStatus.WAITING.value,
                            AgentStateStatus.CANCELLED.value,
                        }
                        else "tools"
                    )
                else:
                    next_route = "agent"
                handle.finish(
                    ok=True,
                    result=AgentComponentResult(
                        next_route=next_route,
                        state_status=state_status,
                    ),
                )
                return patch

            return wrapped

        builder.add_node(
            self.AGENT_NODE,
            traced_node(self.AGENT_NODE, agent_node),
        )
        builder.add_node(
            self.APPROVAL_NODE,
            traced_node(self.APPROVAL_NODE, approval_node),
        )
        builder.add_node(
            self.TOOL_NODE,
            traced_node(self.TOOL_NODE, tool_node),
        )
        builder.add_conditional_edges(
            start_symbol,
            route_from_start,
            {
                "agent": self.AGENT_NODE,
                "approval": self.APPROVAL_NODE,
                "end": end_symbol,
            },
        )
        builder.add_conditional_edges(
            self.AGENT_NODE,
            route_after_agent,
            {
                "approval": self.APPROVAL_NODE,
                "end": end_symbol,
            },
        )
        builder.add_conditional_edges(
            self.APPROVAL_NODE,
            route_after_approval,
            {
                "tools": self.TOOL_NODE,
                "end": end_symbol,
            },
        )
        builder.add_edge(self.TOOL_NODE, self.AGENT_NODE)

        return builder.compile(), initial_state

    def _prepare_resume_initial_state(
        self,
        *,
        db: Session,
        context: ToolExecutionContext,
        thread_id: str,
    ) -> _LangGraphExecutionState:
        if self.recovery_loader is None:
            raise AgentResumeStateError(
                "resume requires a recovery loader"
            )
        if self.checkpoint_writer is None:
            raise AgentResumeStateError(
                "resume requires a checkpoint writer"
            )

        normalized_thread_id = thread_id.strip()
        if not normalized_thread_id:
            raise AgentResumeStateError(
                "thread_id cannot be empty"
            )

        payload = self.recovery_loader.load_resume_checkpoint(
            db,
            thread_id=normalized_thread_id,
            user_id=context.user_id,
            knowledge_base_id=context.knowledge_base_id,
        )
        state = payload.agent_state

        # Recovery Service 已做 DB Scope 校验；Runner 再做一次 trusted context
        # 防御性校验，避免自定义 loader 绕过 Agent Core 边界。
        if state.thread.thread_id != normalized_thread_id:
            raise AgentResumeStateError(
                "checkpoint thread scope does not match request"
            )
        if state.conversation.user_id != context.user_id:
            raise AgentResumeStateError(
                "checkpoint user scope does not match context"
            )
        if (
            state.conversation.knowledge_base_id
            != context.knowledge_base_id
        ):
            raise AgentResumeStateError(
                "checkpoint knowledge base scope does not match context"
            )
        if state.status is not AgentStateStatus.RUNNING:
            raise AgentResumeStateError(
                "checkpoint is not in running state"
            )
        if not state.task:
            raise AgentResumeStateError(
                "resume checkpoint is missing task"
            )

        resumed_agent_state = state.model_copy(
            update={
                "agent_run_id": context.agent_run_id,
                "status": AgentStateStatus.RUNNING,
                "retry_count": state.retry_count + 1,
                "last_error_code": None,
            }
        )

        return {
            "agent_state": resumed_agent_state,
            "history": payload.history,
            "pending_tool_calls": payload.pending_tool_calls,
            "last_model_response": payload.last_model_response,
            # 旧 SSE observation 不在新连接重复发送；Tool 结果已经进入
            # history 的 checkpoint 会直接从 Agent Node 继续。
            "tool_observations": (),
            "final_answer": payload.final_answer,
            "turn": payload.turn,
            "tool_call_count": payload.tool_call_count,
            "seen_tool_call_signatures": (
                payload.seen_tool_call_signatures
            ),
            "pending_approvals": payload.pending_approvals,
            "approved_call_ids": payload.approved_call_ids,
            "rejected_call_ids": payload.rejected_call_ids,
        }

    def _prepare_approved_initial_state(
        self,
        *,
        db: Session,
        context: ToolExecutionContext,
        thread_id: str,
    ) -> _LangGraphExecutionState:
        if self.hitl_loader is None:
            raise AgentApprovalStateError(
                "approval resume requires a HITL loader"
            )
        if self.checkpoint_writer is None:
            raise AgentApprovalStateError(
                "approval resume requires a checkpoint writer"
            )

        normalized_thread_id = thread_id.strip()
        if not normalized_thread_id:
            raise AgentApprovalStateError(
                "thread_id cannot be empty"
            )

        payload = self.hitl_loader.load_approved_checkpoint(
            db,
            thread_id=normalized_thread_id,
            user_id=context.user_id,
            knowledge_base_id=context.knowledge_base_id,
        )
        state = payload.agent_state

        if state.thread.thread_id != normalized_thread_id:
            raise AgentApprovalStateError(
                "approved checkpoint thread scope does not match request"
            )
        if state.conversation.user_id != context.user_id:
            raise AgentApprovalStateError(
                "approved checkpoint user scope does not match context"
            )
        if (
            state.conversation.knowledge_base_id
            != context.knowledge_base_id
        ):
            raise AgentApprovalStateError(
                "approved checkpoint knowledge base scope does not match context"
            )
        if state.status is not AgentStateStatus.WAITING:
            raise AgentApprovalStateError(
                "approved checkpoint is not waiting"
            )
        if not state.task:
            raise AgentApprovalStateError(
                "approved checkpoint is missing task"
            )
        if not payload.pending_tool_calls:
            raise AgentApprovalStateError(
                "approved checkpoint has no pending tool calls"
            )
        required_ids = {
            item.call_id for item in payload.pending_approvals
        }
        if not required_ids:
            raise AgentApprovalStateError(
                "approved checkpoint has no pending approvals"
            )
        if payload.rejected_call_ids:
            raise AgentApprovalStateError(
                "approved checkpoint contains rejected calls"
            )
        if not required_ids.issubset(
            set(payload.approved_call_ids)
        ):
            raise AgentApprovalStateError(
                "not all pending calls are approved"
            )

        resumed_agent_state = state.model_copy(
            update={
                "agent_run_id": context.agent_run_id,
                "status": AgentStateStatus.RUNNING,
                "retry_count": state.retry_count + 1,
                "last_error_code": None,
            }
        )

        return {
            "agent_state": resumed_agent_state,
            "history": payload.history,
            "pending_tool_calls": payload.pending_tool_calls,
            "last_model_response": payload.last_model_response,
            "tool_observations": (),
            "final_answer": payload.final_answer,
            "turn": payload.turn,
            "tool_call_count": payload.tool_call_count,
            "seen_tool_call_signatures": (
                payload.seen_tool_call_signatures
            ),
            "pending_approvals": payload.pending_approvals,
            "approved_call_ids": payload.approved_call_ids,
            "rejected_call_ids": payload.rejected_call_ids,
        }

    def _prepare_initial_state(
        self,
        *,
        context: ToolExecutionContext,
        message: str,
        state: AgentState,
    ) -> _LangGraphExecutionState:
        if state.status is not AgentStateStatus.READY:
            raise ValueError("A6 execution requires a ready AgentState")

        if state.conversation.user_id != context.user_id:
            raise ValueError(
                "AgentState user scope does not match context"
            )

        if (
            state.conversation.knowledge_base_id
            != context.knowledge_base_id
        ):
            raise ValueError(
                "AgentState knowledge base scope does not match context"
            )

        running_state = state.model_copy(
            update={
                "agent_run_id": context.agent_run_id,
                "status": AgentStateStatus.RUNNING,
                "task": message,
                "messages": (
                    *state.messages,
                    ConversationMessagePayload(
                        role=ConversationMessageRole.USER,
                        content=message,
                    ),
                ),
                "last_error_code": None,
            }
        )

        return {
            "agent_state": running_state,
            "history": (),
            "pending_tool_calls": (),
            "last_model_response": None,
            "tool_observations": (),
            "final_answer": None,
            "turn": 0,
            "tool_call_count": 0,
            "seen_tool_call_signatures": (),
            "pending_approvals": (),
            "approved_call_ids": (),
            "rejected_call_ids": (),
        }

    def _save_checkpoint_if_enabled(
        self,
        db: Session,
        graph_state: _LangGraphExecutionState,
        *,
        allowed_previous_statuses: set[AgentStateStatus] | None = None,
        new_execution_attempt: bool = False,
    ) -> None:
        if self.checkpoint_writer is None:
            return

        payload = AgentExecutionCheckpointPayload(
            agent_state=graph_state["agent_state"],
            history=graph_state["history"],
            pending_tool_calls=graph_state["pending_tool_calls"],
            last_model_response=graph_state["last_model_response"],
            tool_observations=graph_state["tool_observations"],
            final_answer=graph_state["final_answer"],
            turn=graph_state["turn"],
            tool_call_count=graph_state["tool_call_count"],
            seen_tool_call_signatures=(
                graph_state["seen_tool_call_signatures"]
            ),
            pending_approvals=graph_state["pending_approvals"],
            approved_call_ids=graph_state["approved_call_ids"],
            rejected_call_ids=graph_state["rejected_call_ids"],
        )
        try:
            self.checkpoint_writer.save_checkpoint(
                db,
                payload,
                allowed_previous_statuses=allowed_previous_statuses,
                new_execution_attempt=new_execution_attempt,
            )
        except AgentCheckpointStateTransitionError as exc:
            run_was_fenced = (
                exc.current_agent_run_id is not None
                and exc.requested_agent_run_id is not None
                and exc.current_agent_run_id != exc.requested_agent_run_id
            )
            if (
                exc.current_status is AgentStateStatus.CANCELLED
                or run_was_fenced
            ):
                raise AgentRunCancellationError(
                    "agent execution was cancelled or superseded"
                ) from exc
            raise

    def _ensure_not_cancelled(
        self,
        *,
        db: Session,
        context: ToolExecutionContext,
        graph_state: _LangGraphExecutionState,
    ) -> None:
        if self.cancellation_probe is None:
            return
        state = graph_state["agent_state"]
        self.cancellation_probe.raise_if_cancelled(
            db,
            thread_id=state.thread.thread_id,
            user_id=context.user_id,
            knowledge_base_id=context.knowledge_base_id,
            agent_run_id=state.agent_run_id,
        )

    @staticmethod
    def _merge_graph_state(
        graph_state: _LangGraphExecutionState,
        patch: Mapping[str, Any],
    ) -> _LangGraphExecutionState:
        merged = dict(graph_state)
        merged.update(patch)
        return LangGraphStatefulRunner._coerce_graph_state(merged)

    @staticmethod
    def _raise_if_interrupted(
        graph_state: _LangGraphExecutionState,
    ) -> None:
        status = graph_state["agent_state"].status
        if status is AgentStateStatus.WAITING:
            raise AgentInterruptRequired(
                thread_id=graph_state["agent_state"].thread.thread_id,
                approvals=graph_state["pending_approvals"],
            )
        if status is AgentStateStatus.CANCELLED:
            raise AgentRunCancellationError(
                "agent execution was cancelled"
            )

    def _graph_config(self) -> dict[str, Any]:
        return {
            # START + Agent/Tool hops need headroom beyond model-turn budget.
            "recursion_limit": self.max_turns * 2 + 4,
            "max_concurrency": 1,
        }

    @staticmethod
    def _coerce_graph_state(
        raw_state: Mapping[str, Any],
    ) -> _LangGraphExecutionState:
        required = {
            "agent_state",
            "history",
            "pending_tool_calls",
            "last_model_response",
            "tool_observations",
            "final_answer",
            "turn",
            "tool_call_count",
            "seen_tool_call_signatures",
            "pending_approvals",
            "approved_call_ids",
            "rejected_call_ids",
        }
        missing = required.difference(raw_state.keys())

        if missing:
            raise RuntimeError(
                "langgraph returned incomplete state: "
                + ", ".join(sorted(missing))
            )

        return dict(raw_state)  # type: ignore[return-value]

    @staticmethod
    def _load_langgraph_components() -> tuple[
        Callable[[type[_LangGraphExecutionState]], _StateGraphBuilder],
        Any,
        Any,
    ]:
        """延迟加载 LangGraph，避免 Native/LangChain import 被反向耦合。"""

        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError as exc:  # pragma: no cover - depends on local env
            raise RuntimeError(
                "LangGraph Stateful Runtime requires langgraph>=1.0,<2.0"
            ) from exc

        return StateGraph, START, END

    @staticmethod
    def _close_iterator(
        iterator: Iterator[Any] | None,
    ) -> None:
        if iterator is None:
            return

        close = getattr(iterator, "close", None)
        if callable(close):
            close()
