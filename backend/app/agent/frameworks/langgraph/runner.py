"""v2.3-A6 LangGraph Stateful Runtime with durable checkpoint hooks.

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
    AgentCheckpointWriter,
    AgentExecutionCheckpointPayload,
)
from app.agent.context import ToolExecutionContext
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
        source: str,
        path: Callable[..., Any],
        path_map: Mapping[str, Any],
    ) -> Any:
        ...

    def compile(self) -> _CompiledGraph:
        ...


class _LangGraphExecutionState(TypedDict):
    """
    A5 的请求内 Graph State。

    ``agent_state`` 是 A1 冻结的框架无关状态；其余字段是本次编排需要的
    transient data。A6 接入 Checkpoint 时，再决定哪些 transient 字段必须
    晋升为可持久化 State Contract，A5 不提前把框架细节塞进 AgentState。
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
    v2.3-A6 显式 StateGraph + durable checkpoint Candidate。

    只替换 Agent Loop 的编排方式：

        START -> Agent Node -> needs tools?
                          | yes -> Tool Node -> Agent Node
                          | no  -> END

    Tool Contract、ToolDispatcher、Trusted Context、安全 ToolError 回填、
    max_turns / max_tool_calls / timeout / repeated-call protection 全部沿用
    Native Baseline，避免为了“上 LangGraph”重写既有业务边界。

    A6 通过框架无关 checkpoint writer 在关键 Node 边界落库；
    本版本只完成持久化与读取，不自动从 checkpoint 续跑。Resume / HITL
    留给后续小版本。
    """

    RUNNER_VERSION = "0.1.0"
    GRAPH_VERSION = "1.0"

    AGENT_NODE = "agent"
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
    ) -> None:
        super().__init__(
            llm_service=llm_service,
            tools=tools,
            max_turns=max_turns,
            max_tool_calls=max_tool_calls,
            max_duration_seconds=max_duration_seconds,
        )
        self.checkpoint_writer = checkpoint_writer

    def run(
        self,
        *,
        db: Session,
        context: ToolExecutionContext,
        message: str,
        state: AgentState,
        observer: AgentRunObserver | None = None,
    ) -> LangGraphStatefulResult:
        """执行一次 Minimal StateGraph，并返回最终框架无关 AgentState。"""

        graph, initial_state = self._build_graph_execution(
            db=db,
            context=context,
            message=message,
            state=state,
            observer=observer,
        )
        final_raw = graph.invoke(
            initial_state,
            config=self._graph_config(),
        )
        final_state = self._coerce_graph_state(final_raw)
        answer = (final_state["final_answer"] or "").strip()

        if not answer:
            raise RuntimeError("langgraph agent completed without final answer")

        return LangGraphStatefulResult(
            answer=answer,
            turns=final_state["turn"],
            tool_call_count=final_state["tool_call_count"],
            state=final_state["agent_state"],
        )

    def run_events(
        self,
        *,
        db: Session,
        context: ToolExecutionContext,
        message: str,
        state: AgentState,
        observer: AgentRunObserver | None = None,
    ) -> Iterator[AgentRunEvent]:
        """执行 Minimal StateGraph，并继续输出现有 provider-neutral 安全事件。"""

        graph, initial_state = self._build_graph_execution(
            db=db,
            context=context,
            message=message,
            state=state,
            observer=observer,
        )
        graph_stream: Iterator[Mapping[str, Any]] | None = None
        completed = False

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
    ) -> tuple[_CompiledGraph, _LangGraphExecutionState]:
        normalized_message = self._normalize_message(message)
        initial_state = self._prepare_initial_state(
            context=context,
            message=normalized_message,
            state=state,
        )
        self._save_checkpoint_if_enabled(db, initial_state)
        started_at = time.monotonic()

        def agent_node(
            graph_state: _LangGraphExecutionState,
        ) -> dict[str, Any]:
            self._ensure_within_deadline(started_at)

            turn = graph_state["turn"] + 1
            if turn > self.max_turns:
                raise AgentTurnLimitError("agent exceeded max_turns")

            response = self.llm_service.chat_with_tool_history(
                message=(
                    graph_state["agent_state"].task
                    or normalized_message
                ),
                tool_contracts=self.tool_contracts,
                history=graph_state["history"],
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

        def tool_node(
            graph_state: _LangGraphExecutionState,
        ) -> dict[str, Any]:
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
                tool_started_at = time.perf_counter()
                outcome = self._execute_tool_call(
                    db=db,
                    context=context,
                    tool_call=tool_call,
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
            }
            self._save_checkpoint_if_enabled(
                db,
                self._merge_graph_state(graph_state, patch),
            )
            return patch

        def route_after_agent(
            graph_state: _LangGraphExecutionState,
        ) -> str:
            if graph_state["pending_tool_calls"]:
                return "tools"
            return "end"

        state_graph_factory, start_symbol, end_symbol = (
            self._load_langgraph_components()
        )
        builder: _StateGraphBuilder = state_graph_factory(
            _LangGraphExecutionState
        )
        builder.add_node(self.AGENT_NODE, agent_node)
        builder.add_node(self.TOOL_NODE, tool_node)
        builder.add_edge(start_symbol, self.AGENT_NODE)
        builder.add_conditional_edges(
            self.AGENT_NODE,
            route_after_agent,
            {
                "tools": self.TOOL_NODE,
                "end": end_symbol,
            },
        )
        builder.add_edge(self.TOOL_NODE, self.AGENT_NODE)

        return builder.compile(), initial_state

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
        }

    def _save_checkpoint_if_enabled(
        self,
        db: Session,
        graph_state: _LangGraphExecutionState,
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
        )
        self.checkpoint_writer.save_checkpoint(db, payload)

    @staticmethod
    def _merge_graph_state(
        graph_state: _LangGraphExecutionState,
        patch: Mapping[str, Any],
    ) -> _LangGraphExecutionState:
        merged = dict(graph_state)
        merged.update(patch)
        return LangGraphStatefulRunner._coerce_graph_state(merged)

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
