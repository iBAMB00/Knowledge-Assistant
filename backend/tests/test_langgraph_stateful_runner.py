from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from app.agent.context import ToolExecutionContext
from app.agent.frameworks.langgraph.runner import (
    LangGraphStatefulRunner,
)
from app.agent.model_response import (
    LLMToolCall,
    LLMToolExchange,
    LLMToolResponse,
)
from app.agent.native_agent import AgentRepeatedToolCallError
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
from app.schemas.conversation_contract import ConversationScope


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
        source: str,
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

    def save_checkpoint(self, db: Session, payload):
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
