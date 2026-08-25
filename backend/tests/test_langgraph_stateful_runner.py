from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from app.agent.context import ToolExecutionContext
from app.agent.hitl import (
    AgentApprovalStateError,
    AgentInterruptRequired,
    ToolNameApprovalPolicy,
)
from app.agent.frameworks.langgraph.runner import (
    LangGraphStatefulRunner,
)
from app.agent.checkpoint import AgentExecutionCheckpointPayload, AgentResumeStateError
from app.agent.model_response import (
    LLMToolCall,
    LLMToolExchange,
    LLMToolResponse,
    LLMToolResult,
)
from app.agent.native_agent import AgentRepeatedToolCallError
from app.agent.run_control import AgentRunCancellationError
from app.agent.run_event import (
    AgentMessageEvent,
    AgentStatusEvent,
    AgentToolCallEvent,
    AgentToolResultEvent,
)
from app.agent.state import AgentState, AgentThreadIdentity
from app.agent.tools.base import BaseAgentTool, ToolContract, ToolRiskLevel
from app.constants.agent_state_status import AgentStateStatus
from app.constants.conversation_message_role import (
    ConversationMessageRole,
)
from app.constants.conversation_mode import ConversationMode
from app.constants.user_role import UserRole
from app.schemas.conversation_contract import (
    ConversationMessagePayload,
    ConversationScope,
)


class EchoInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str


class EchoOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    echoed: str


class EchoTool(BaseAgentTool[EchoInput, EchoOutput]):
    name = "echo"
    version = "1.0"
    description = "Echo text"
    risk_level = ToolRiskLevel.READ_ONLY
    input_model = EchoInput
    output_model = EchoOutput

    def execute(
        self,
        db: Session,
        context: ToolExecutionContext,
        tool_input: EchoInput,
    ) -> EchoOutput:
        return EchoOutput(echoed=tool_input.text)


class ScriptedLLM:
    def __init__(
        self,
        responses: list[LLMToolResponse],
    ) -> None:
        self.responses = list(responses)
        self.received_histories: list[
            tuple[LLMToolExchange, ...]
        ] = []

    def chat_with_tool_history(
        self,
        *,
        message: str,
        tool_contracts: Sequence[ToolContract],
        history: Sequence[LLMToolExchange],
    ) -> LLMToolResponse:
        self.received_histories.append(tuple(history))

        if not self.responses:
            raise AssertionError("unexpected model call")

        return self.responses.pop(0)


class FakeCompiledGraph:
    def __init__(
        self,
        *,
        nodes: dict[
            str,
            Callable[[dict[str, Any]], dict[str, Any]],
        ],
        edges: dict[Any, Any],
        conditional: dict[
            str,
            tuple[
                Callable[[dict[str, Any]], str],
                Mapping[str, Any],
            ],
        ],
        start: Any,
        end: Any,
    ) -> None:
        self.nodes = nodes
        self.edges = edges
        self.conditional = conditional
        self.start = start
        self.end = end

    def _steps(
        self,
        input_state: Mapping[str, Any],
    ) -> Iterator[
        tuple[str, dict[str, Any], dict[str, Any]]
    ]:
        state = dict(input_state)
        if self.start in self.conditional:
            route, path_map = self.conditional[self.start]
            current = path_map[route(state)]
        else:
            current = self.edges[self.start]
        guard = 0

        while current != self.end:
            guard += 1
            if guard > 50:
                raise RuntimeError("fake graph loop")

            patch = self.nodes[current](state)
            state.update(patch)
            yield current, patch, dict(state)

            if current in self.conditional:
                route, path_map = self.conditional[current]
                current = path_map[route(state)]
            else:
                current = self.edges[current]

    def invoke(
        self,
        input: Mapping[str, Any],
        config: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        final = dict(input)

        for _, _, final in self._steps(input):
            pass

        return final

    def stream(
        self,
        input: Mapping[str, Any],
        config: Mapping[str, Any] | None = None,
        *,
        stream_mode: str,
    ) -> Iterator[Mapping[str, Any]]:
        assert stream_mode == "updates"

        for node, patch, _ in self._steps(input):
            yield {node: patch}


class FakeStateGraph:
    START = "__start__"
    END = "__end__"

    def __init__(self, _schema: type[Any]) -> None:
        self.nodes: dict[
            str,
            Callable[[dict[str, Any]], dict[str, Any]],
        ] = {}
        self.edges: dict[Any, Any] = {}
        self.conditional: dict[
            str,
            tuple[
                Callable[[dict[str, Any]], str],
                Mapping[str, Any],
            ],
        ] = {}

    def add_node(
        self,
        node: str,
        action: Callable[
            [dict[str, Any]],
            dict[str, Any],
        ],
    ) -> None:
        self.nodes[node] = action

    def add_edge(
        self,
        start_key: Any,
        end_key: Any,
    ) -> None:
        self.edges[start_key] = end_key

    def add_conditional_edges(
        self,
        source: Any,
        path: Callable[[dict[str, Any]], str],
        path_map: Mapping[str, Any],
    ) -> None:
        self.conditional[source] = (path, path_map)

    def compile(self) -> FakeCompiledGraph:
        return FakeCompiledGraph(
            nodes=self.nodes,
            edges=self.edges,
            conditional=self.conditional,
            start=self.START,
            end=self.END,
        )


def build_state() -> AgentState:
    return AgentState(
        conversation=ConversationScope(
            conversation_id=101,
            user_id=7,
            mode=ConversationMode.AGENT,
            knowledge_base_id=11,
        ),
        thread=AgentThreadIdentity(
            thread_id="conversation:101",
            conversation_id=101,
        ),
    )


def build_context() -> ToolExecutionContext:
    return ToolExecutionContext(
        user_id=7,
        role=UserRole.USER,
        knowledge_base_id=11,
        request_id="langgraph-a5-test",
        agent_run_id=88,
    )


def build_runner(
    responses: list[LLMToolResponse],
) -> tuple[LangGraphStatefulRunner, ScriptedLLM]:
    llm = ScriptedLLM(responses)
    runner = LangGraphStatefulRunner(
        llm_service=llm,
        tools=[EchoTool()],
        max_turns=4,
        max_tool_calls=4,
    )
    runner._load_langgraph_components = lambda: (  # type: ignore[method-assign]
        FakeStateGraph,
        FakeStateGraph.START,
        FakeStateGraph.END,
    )
    return runner, llm


def test_minimal_graph_direct_answer_updates_agent_state(
    db: Session,
) -> None:
    runner, _ = build_runner(
        [LLMToolResponse(content="直接回答")]
    )

    result = runner.run(
        db=db,
        context=build_context(),
        message="你好",
        state=build_state(),
    )

    assert result.answer == "直接回答"
    assert result.turns == 1
    assert result.tool_call_count == 0
    assert result.state.status is AgentStateStatus.SUCCEEDED
    assert result.state.agent_run_id == 88
    assert result.state.task == "你好"
    assert [
        item.role
        for item in result.state.messages
    ] == [
        ConversationMessageRole.USER,
        ConversationMessageRole.ASSISTANT,
    ]
    assert [
        item.content
        for item in result.state.messages
    ] == [
        "你好",
        "直接回答",
    ]


def test_minimal_graph_routes_agent_tool_agent(
    db: Session,
) -> None:
    runner, llm = build_runner(
        [
            LLMToolResponse(
                tool_calls=[
                    LLMToolCall(
                        id="call-1",
                        name="echo",
                        arguments_json='{"text":"hello"}',
                    )
                ]
            ),
            LLMToolResponse(content="工具执行完成"),
        ]
    )

    result = runner.run(
        db=db,
        context=build_context(),
        message="调用 echo",
        state=build_state(),
    )

    assert result.answer == "工具执行完成"
    assert result.turns == 2
    assert result.tool_call_count == 1
    assert len(llm.received_histories) == 2
    assert llm.received_histories[0] == ()
    assert len(llm.received_histories[1]) == 1

    exchange = llm.received_histories[1][0]
    assert exchange.tool_results[0].tool_name == "echo"
    assert (
        '"echoed":"hello"'
        in exchange.tool_results[0].content_json
    )


def test_minimal_graph_stream_reuses_safe_agent_events(
    db: Session,
) -> None:
    runner, _ = build_runner(
        [
            LLMToolResponse(
                tool_calls=[
                    LLMToolCall(
                        id="call-1",
                        name="echo",
                        arguments_json='{"text":"hello"}',
                    )
                ]
            ),
            LLMToolResponse(content="完成"),
        ]
    )

    events = list(
        runner.run_events(
            db=db,
            context=build_context(),
            message="调用工具",
            state=build_state(),
        )
    )

    assert [type(event) for event in events] == [
        AgentStatusEvent,
        AgentToolCallEvent,
        AgentToolResultEvent,
        AgentStatusEvent,
        AgentMessageEvent,
    ]
    assert events[1].tool_name == "echo"
    assert events[2].ok is True
    assert events[-1].content == "完成"


def test_minimal_graph_rejects_state_scope_mismatch(
    db: Session,
) -> None:
    runner, _ = build_runner(
        [LLMToolResponse(content="不会执行")]
    )
    wrong_context = build_context().model_copy(
        update={"user_id": 8}
    )

    with pytest.raises(ValueError, match="user scope"):
        runner.run(
            db=db,
            context=wrong_context,
            message="测试",
            state=build_state(),
        )


def test_minimal_graph_keeps_repeated_tool_call_protection(
    db: Session,
) -> None:
    runner, _ = build_runner(
        [
            LLMToolResponse(
                tool_calls=[
                    LLMToolCall(
                        id="call-1",
                        name="echo",
                        arguments_json='{"text":"same"}',
                    )
                ]
            ),
            LLMToolResponse(
                tool_calls=[
                    LLMToolCall(
                        id="call-2",
                        name="echo",
                        arguments_json='{"text":"same"}',
                    )
                ]
            ),
        ]
    )

    with pytest.raises(AgentRepeatedToolCallError):
        runner.run(
            db=db,
            context=build_context(),
            message="重复调用",
            state=build_state(),
        )


class RecordingCheckpointWriter:
    def __init__(self) -> None:
        self.payloads = []

    def save_checkpoint(
        self,
        db: Session,
        payload,
        *,
        allowed_previous_statuses=None,
        new_execution_attempt=False,
    ):
        self.payloads.append(payload)
        return object()


def test_minimal_graph_emits_durable_checkpoint_boundaries(db: Session) -> None:
    llm = ScriptedLLM(
        [
            LLMToolResponse(
                tool_calls=[
                    LLMToolCall(
                        id="checkpoint-call-1",
                        name="echo",
                        arguments_json='{"text":"persist"}',
                    )
                ]
            ),
            LLMToolResponse(content="checkpoint done"),
        ]
    )
    writer = RecordingCheckpointWriter()
    runner = LangGraphStatefulRunner(
        llm_service=llm,
        tools=[EchoTool()],
        checkpoint_writer=writer,
    )
    runner._load_langgraph_components = lambda: (  # type: ignore[method-assign]
        FakeStateGraph,
        FakeStateGraph.START,
        FakeStateGraph.END,
    )

    result = runner.run(
        db=db,
        context=build_context(),
        message="checkpoint test",
        state=build_state(),
    )

    assert result.answer == "checkpoint done"
    # initial -> model(tool call) -> tool result -> final model
    assert len(writer.payloads) == 4
    assert writer.payloads[0].turn == 0
    assert writer.payloads[1].pending_tool_calls[0].name == "echo"
    assert len(writer.payloads[2].history) == 1
    assert writer.payloads[-1].agent_state.status is AgentStateStatus.SUCCEEDED
    assert writer.payloads[-1].final_answer == "checkpoint done"


class CountingEchoTool(EchoTool):
    def __init__(self) -> None:
        self.call_count = 0

    def execute(
        self,
        db: Session,
        context: ToolExecutionContext,
        tool_input: EchoInput,
    ) -> EchoOutput:
        self.call_count += 1
        return super().execute(db, context, tool_input)


class StaticRecoveryLoader:
    def __init__(self, payload: AgentExecutionCheckpointPayload) -> None:
        self.payload = payload
        self.requests: list[tuple[str, int, int]] = []

    def load_resume_checkpoint(
        self,
        db: Session,
        *,
        thread_id: str,
        user_id: int,
        knowledge_base_id: int,
    ) -> AgentExecutionCheckpointPayload:
        self.requests.append(
            (thread_id, user_id, knowledge_base_id)
        )
        return self.payload


def build_running_checkpoint(
    *,
    pending: bool,
) -> AgentExecutionCheckpointPayload:
    tool_call = LLMToolCall(
        id="resume-call-1",
        name="echo",
        arguments_json='{"text":"persist"}',
    )
    model_response = LLMToolResponse(tool_calls=[tool_call])
    state = build_state().model_copy(
        update={
            "agent_run_id": 77,
            "status": AgentStateStatus.RUNNING,
            "task": "恢复执行",
            "messages": (
                ConversationMessagePayload(
                    role=ConversationMessageRole.USER,
                    content="恢复执行",
                ),
            ),
        }
    )

    history: tuple[LLMToolExchange, ...] = ()
    pending_calls: tuple[LLMToolCall, ...] = (tool_call,)
    if not pending:
        history = (
            LLMToolExchange(
                response=model_response,
                tool_results=[
                    LLMToolResult(
                        call_id=tool_call.id,
                        tool_name=tool_call.name,
                        content_json='{"echoed":"persist"}',
                    )
                ],
            ),
        )
        pending_calls = ()

    return AgentExecutionCheckpointPayload(
        agent_state=state,
        history=history,
        pending_tool_calls=pending_calls,
        last_model_response=model_response,
        turn=1,
        tool_call_count=1,
        seen_tool_call_signatures=(
            'echo:{"text":"persist"}',
        ),
    )


def build_resume_runner(
    *,
    payload: AgentExecutionCheckpointPayload,
    responses: list[LLMToolResponse],
) -> tuple[
    LangGraphStatefulRunner,
    ScriptedLLM,
    CountingEchoTool,
    RecordingCheckpointWriter,
]:
    llm = ScriptedLLM(responses)
    tool = CountingEchoTool()
    writer = RecordingCheckpointWriter()
    runner = LangGraphStatefulRunner(
        llm_service=llm,
        tools=[tool],
        checkpoint_writer=writer,
        recovery_loader=StaticRecoveryLoader(payload),
    )
    runner._load_langgraph_components = lambda: (  # type: ignore[method-assign]
        FakeStateGraph,
        FakeStateGraph.START,
        FakeStateGraph.END,
    )
    return runner, llm, tool, writer


def test_resume_after_durable_tool_result_does_not_execute_tool_again(
    db: Session,
) -> None:
    runner, llm, tool, writer = build_resume_runner(
        payload=build_running_checkpoint(pending=False),
        responses=[LLMToolResponse(content="从已保存结果继续完成")],
    )

    result = runner.resume(
        db=db,
        context=build_context(),
        thread_id="conversation:101",
    )

    assert result.answer == "从已保存结果继续完成"
    assert result.turns == 2
    assert result.tool_call_count == 1
    assert tool.call_count == 0
    assert len(llm.received_histories) == 1
    assert len(llm.received_histories[0]) == 1
    assert result.state.agent_run_id == 88
    assert result.state.retry_count == 1
    assert result.state.status is AgentStateStatus.SUCCEEDED
    # resume-attempt boundary + final answer
    assert len(writer.payloads) == 2
    assert writer.payloads[0].agent_state.retry_count == 1


def test_resume_from_pending_tool_call_starts_at_tool_node_once(
    db: Session,
) -> None:
    runner, llm, tool, writer = build_resume_runner(
        payload=build_running_checkpoint(pending=True),
        responses=[LLMToolResponse(content="工具恢复后完成")],
    )

    result = runner.resume(
        db=db,
        context=build_context(),
        thread_id="conversation:101",
    )

    assert result.answer == "工具恢复后完成"
    assert result.turns == 2
    assert result.tool_call_count == 1
    assert tool.call_count == 1
    assert len(llm.received_histories) == 1
    assert len(llm.received_histories[0]) == 1
    # resume boundary -> tool result -> final model
    assert len(writer.payloads) == 3
    assert writer.payloads[1].pending_tool_calls == ()
    assert len(writer.payloads[1].history) == 1


def test_resume_events_replays_safe_pending_tool_metadata(
    db: Session,
) -> None:
    runner, _, tool, _ = build_resume_runner(
        payload=build_running_checkpoint(pending=True),
        responses=[LLMToolResponse(content="恢复完成")],
    )

    events = list(
        runner.resume_events(
            db=db,
            context=build_context(),
            thread_id="conversation:101",
        )
    )

    assert [type(event) for event in events] == [
        AgentToolCallEvent,
        AgentToolResultEvent,
        AgentStatusEvent,
        AgentMessageEvent,
    ]
    assert events[0].tool_name == "echo"
    assert events[1].ok is True
    assert events[-1].content == "恢复完成"
    assert tool.call_count == 1


def test_resume_requires_recovery_loader(db: Session) -> None:
    runner, _ = build_runner([LLMToolResponse(content="不会执行")])

    with pytest.raises(AgentResumeStateError, match="recovery loader"):
        runner.resume(
            db=db,
            context=build_context(),
            thread_id="conversation:101",
        )


class StaticHITLLoader:
    def __init__(self, payload: AgentExecutionCheckpointPayload) -> None:
        self.payload = payload
        self.requests: list[tuple[str, int, int]] = []

    def load_approved_checkpoint(
        self,
        db: Session,
        *,
        thread_id: str,
        user_id: int,
        knowledge_base_id: int,
    ) -> AgentExecutionCheckpointPayload:
        self.requests.append((thread_id, user_id, knowledge_base_id))
        return self.payload


def test_hitl_interrupts_before_tool_and_persists_waiting_checkpoint(
    db: Session,
) -> None:
    llm = ScriptedLLM(
        [
            LLMToolResponse(
                tool_calls=[
                    LLMToolCall(
                        id="approval-call-1",
                        name="echo",
                        arguments_json='{"text":"sensitive"}',
                    )
                ]
            )
        ]
    )
    tool = CountingEchoTool()
    writer = RecordingCheckpointWriter()
    runner = LangGraphStatefulRunner(
        llm_service=llm,
        tools=[tool],
        checkpoint_writer=writer,
        interrupt_policy=ToolNameApprovalPolicy(
            {"echo": "测试 Tool 需要人工确认"}
        ),
    )
    runner._load_langgraph_components = lambda: (  # type: ignore[method-assign]
        FakeStateGraph,
        FakeStateGraph.START,
        FakeStateGraph.END,
    )

    with pytest.raises(AgentInterruptRequired) as exc_info:
        runner.run(
            db=db,
            context=build_context(),
            message="执行需要审批的 Tool",
            state=build_state(),
        )

    assert tool.call_count == 0
    assert exc_info.value.thread_id == "conversation:101"
    assert len(exc_info.value.approvals) == 1
    assert exc_info.value.approvals[0].tool_name == "echo"
    assert exc_info.value.approvals[0].reason == "测试 Tool 需要人工确认"
    # arguments_json 不进入公开审批元数据。
    assert not hasattr(exc_info.value.approvals[0], "arguments_json")

    waiting = writer.payloads[-1]
    assert waiting.agent_state.status is AgentStateStatus.WAITING
    assert waiting.pending_tool_calls[0].id == "approval-call-1"
    assert waiting.pending_approvals[0].call_id == "approval-call-1"
    assert waiting.approved_call_ids == ()
    # initial -> model(tool call) -> WAITING approval checkpoint
    assert len(writer.payloads) == 3


def test_resume_after_approval_executes_pending_tool_once(
    db: Session,
) -> None:
    tool_call = LLMToolCall(
        id="approval-call-1",
        name="echo",
        arguments_json='{"text":"approved"}',
    )
    model_response = LLMToolResponse(tool_calls=[tool_call])
    waiting_state = build_state().model_copy(
        update={
            "agent_run_id": 77,
            "status": AgentStateStatus.WAITING,
            "task": "执行审批 Tool",
            "messages": (
                ConversationMessagePayload(
                    role=ConversationMessageRole.USER,
                    content="执行审批 Tool",
                ),
            ),
        }
    )
    requirement_policy = ToolNameApprovalPolicy(
        {"echo": "测试 Tool 需要人工确认"}
    )
    requirement = requirement_policy.get_approval_requirement(
        tool_call=tool_call,
        tool_contract=EchoTool().get_contract(),
    )
    assert requirement is not None

    approved_payload = AgentExecutionCheckpointPayload(
        agent_state=waiting_state,
        pending_tool_calls=(tool_call,),
        last_model_response=model_response,
        turn=1,
        tool_call_count=1,
        seen_tool_call_signatures=('echo:{"text":"approved"}',),
        pending_approvals=(requirement,),
        approved_call_ids=(tool_call.id,),
    )

    llm = ScriptedLLM(
        [LLMToolResponse(content="审批后执行完成")]
    )
    tool = CountingEchoTool()
    writer = RecordingCheckpointWriter()
    runner = LangGraphStatefulRunner(
        llm_service=llm,
        tools=[tool],
        checkpoint_writer=writer,
        interrupt_policy=requirement_policy,
        hitl_loader=StaticHITLLoader(approved_payload),
    )
    runner._load_langgraph_components = lambda: (  # type: ignore[method-assign]
        FakeStateGraph,
        FakeStateGraph.START,
        FakeStateGraph.END,
    )

    result = runner.resume_after_approval(
        db=db,
        context=build_context(),
        thread_id="conversation:101",
    )

    assert result.answer == "审批后执行完成"
    assert result.state.status is AgentStateStatus.SUCCEEDED
    assert result.state.retry_count == 1
    assert tool.call_count == 1
    assert len(llm.received_histories) == 1
    assert len(llm.received_histories[0]) == 1
    # approval-resume boundary -> tool result -> final answer
    assert len(writer.payloads) == 3
    assert writer.payloads[0].approved_call_ids == ("approval-call-1",)
    assert writer.payloads[1].pending_tool_calls == ()
    assert writer.payloads[1].pending_approvals == ()


def test_resume_after_approval_requires_hitl_loader(db: Session) -> None:
    runner, _ = build_runner([LLMToolResponse(content="不会执行")])

    with pytest.raises(AgentApprovalStateError, match="HITL loader"):
        runner.resume_after_approval(
            db=db,
            context=build_context(),
            thread_id="conversation:101",
        )


def test_resume_after_approval_events_replays_safe_tool_metadata(
    db: Session,
) -> None:
    tool_call = LLMToolCall(
        id="approval-event-call",
        name="echo",
        arguments_json='{"text":"approved-event"}',
    )
    response = LLMToolResponse(tool_calls=[tool_call])
    waiting_state = build_state().model_copy(
        update={
            "agent_run_id": 77,
            "status": AgentStateStatus.WAITING,
            "task": "审批后流式执行",
            "messages": (
                ConversationMessagePayload(
                    role=ConversationMessageRole.USER,
                    content="审批后流式执行",
                ),
            ),
        }
    )
    policy = ToolNameApprovalPolicy(
        {"echo": "测试 Tool 需要人工确认"}
    )
    requirement = policy.get_approval_requirement(
        tool_call=tool_call,
        tool_contract=EchoTool().get_contract(),
    )
    assert requirement is not None
    payload = AgentExecutionCheckpointPayload(
        agent_state=waiting_state,
        pending_tool_calls=(tool_call,),
        last_model_response=response,
        turn=1,
        tool_call_count=1,
        seen_tool_call_signatures=(
            'echo:{"text":"approved-event"}',
        ),
        pending_approvals=(requirement,),
        approved_call_ids=(tool_call.id,),
    )

    llm = ScriptedLLM([LLMToolResponse(content="审批流式完成")])
    tool = CountingEchoTool()
    runner = LangGraphStatefulRunner(
        llm_service=llm,
        tools=[tool],
        checkpoint_writer=RecordingCheckpointWriter(),
        interrupt_policy=policy,
        hitl_loader=StaticHITLLoader(payload),
    )
    runner._load_langgraph_components = lambda: (  # type: ignore[method-assign]
        FakeStateGraph,
        FakeStateGraph.START,
        FakeStateGraph.END,
    )

    events = list(
        runner.resume_after_approval_events(
            db=db,
            context=build_context(),
            thread_id="conversation:101",
        )
    )

    assert [type(event) for event in events] == [
        AgentToolCallEvent,
        AgentToolResultEvent,
        AgentStatusEvent,
        AgentMessageEvent,
    ]
    assert events[0].tool_name == "echo"
    assert not hasattr(events[0], "arguments_json")
    assert events[-1].content == "审批流式完成"
    assert tool.call_count == 1


class _CancelBeforeToolProbe:
    def __init__(self) -> None:
        self.calls = 0

    def raise_if_cancelled(
        self,
        db: Session,
        *,
        thread_id: str,
        user_id: int,
        knowledge_base_id: int,
        agent_run_id: int | str | None = None,
    ) -> None:
        self.calls += 1
        # agent start, model return, approval start 之后，在 tool_node 入口取消。
        if self.calls >= 4:
            raise AgentRunCancellationError(
                "agent execution was cancelled"
            )


class _CountingEchoTool(EchoTool):
    def __init__(self) -> None:
        self.execute_count = 0

    def execute(
        self,
        db: Session,
        context: ToolExecutionContext,
        tool_input: EchoInput,
    ) -> EchoOutput:
        self.execute_count += 1
        return super().execute(db, context, tool_input)


def test_langgraph_cancellation_probe_stops_pending_tool_before_execution(
    db: Session,
) -> None:
    llm = ScriptedLLM(
        [
            LLMToolResponse(
                tool_calls=[
                    LLMToolCall(
                        id="cancel-call",
                        name="echo",
                        arguments_json='{"text":"hello"}',
                    )
                ]
            )
        ]
    )
    tool = _CountingEchoTool()
    probe = _CancelBeforeToolProbe()
    runner = LangGraphStatefulRunner(
        llm_service=llm,
        tools=[tool],
        cancellation_probe=probe,
    )
    runner._load_langgraph_components = lambda: (  # type: ignore[method-assign]
        FakeStateGraph,
        FakeStateGraph.START,
        FakeStateGraph.END,
    )

    with pytest.raises(AgentRunCancellationError, match="cancelled"):
        runner.run(
            db=db,
            context=build_context(),
            message="调用 echo",
            state=build_state(),
        )

    assert tool.execute_count == 0


def test_a4_graph_node_trace_records_fresh_route_without_state_payload(
    db: Session,
) -> None:
    from dataclasses import dataclass, field

    from app.agent.observability import (
        AgentComponentCallContext,
        AgentComponentResult,
        AgentComponentTracer,
        AgentGraphExecutionMode,
        AgentObservationKind,
        AgentTraceContext,
    )
    from app.constants.agent_runtime import AgentRuntime

    @dataclass
    class Handle:
        span_id: str
        finishes: list[dict[str, Any]] = field(default_factory=list)

        def finish(self, *, ok=True, result=None, error_code=None):
            self.finishes.append(
                {"ok": ok, "result": result, "error_code": error_code}
            )

    @dataclass
    class TraceHandle:
        trace_id: str
        provider_trace_id: str | None = None
        contexts: list[AgentComponentCallContext] = field(default_factory=list)
        handles: list[Handle] = field(default_factory=list)

        def start_component_call(self, *, call_context):
            self.contexts.append(call_context)
            handle = Handle(call_context.span.span_id)
            self.handles.append(handle)
            return handle

        def start_model_call(self, **kwargs):
            raise AssertionError("not used")

        def finish(self, **kwargs):
            return None

    trace_context = AgentTraceContext(
        trace_id="2" * 32,
        request_id="graph-a4",
        runtime=AgentRuntime.LANGGRAPH,
        user_id=7,
        knowledge_base_id=11,
        agent_run_id=88,
        agent_version="agent-v2.5",
        prompt_id="agent.tool-calling-system",
        prompt_version="1.1.0",
        toolset_version="toolset-v2:test",
        retrieval_config_version="retrieval-v1:test",
    )
    trace_handle = TraceHandle(trace_id=trace_context.trace_id)
    tracer = AgentComponentTracer(
        trace_context=trace_context,
        trace_handle=trace_handle,  # type: ignore[arg-type]
    )
    runner, _ = build_runner([LLMToolResponse(content="done")])

    result = runner.run(
        db=db,
        context=build_context(),
        message="hello",
        state=build_state(),
        component_tracer=tracer,
    )

    assert result.answer == "done"
    assert len(trace_handle.contexts) == 1
    graph_context = trace_handle.contexts[0]
    assert graph_context.span.kind is AgentObservationKind.GRAPH_NODE
    assert graph_context.graph_node == runner.AGENT_NODE
    assert graph_context.graph_execution_mode is AgentGraphExecutionMode.FRESH
    finish = trace_handle.handles[0].finishes[0]
    assert finish["result"] == AgentComponentResult(
        next_route="end",
        state_status="succeeded",
    )
