import json
import sys
import types
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.agent.agent_prompt import (
    AGENT_TOOL_CALLING_PROMPT_VERSION,
    build_agent_tool_calling_system_prompt,
)
from app.agent.context import ToolExecutionContext
from app.agent.frameworks.langchain.model_adapter import LangChainModelAdapter
from app.agent.frameworks.langchain.runner import (
    LangChainAgentError,
    LangChainAgentLimitError,
    LangChainSingleAgentRunner,
)
from app.agent.tools.base import BaseAgentTool, ToolRiskLevel
from app.constants.user_role import UserRole
from app.services.llm_service import LLMService


class DemoInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)


class DemoOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str


class DemoTool(BaseAgentTool[DemoInput, DemoOutput]):
    name = "demo_search"
    version = "1.0.0"
    description = "Search deterministic demo data."
    risk_level = ToolRiskLevel.READ_ONLY
    input_model = DemoInput
    output_model = DemoOutput

    def __init__(self) -> None:
        self.last_context: ToolExecutionContext | None = None

    def execute(self, db, context, tool_input):
        self.last_context = context
        return DemoOutput(value=f"result:{tool_input.query}")


class KnowledgeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    result_count: int
    items: list[dict[str, Any]]


class KnowledgeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str


class KnowledgeTool(BaseAgentTool[KnowledgeInput, KnowledgeOutput]):
    name = "search_knowledge"
    version = "1.1.0"
    description = "Search current knowledge base."
    risk_level = ToolRiskLevel.READ_ONLY
    input_model = KnowledgeInput
    output_model = KnowledgeOutput

    def execute(self, db, context, tool_input):
        return KnowledgeOutput(
            result_count=1,
            items=[
                {
                    "document_id": 1,
                    "chunk_id": 2,
                    "content": "Qdrant uses port 6333.",
                    "source_ref": "doc:1:chunk:2",
                }
            ],
        )

    def extract_evidence_refs(self, output: KnowledgeOutput) -> list[str]:
        return [item["source_ref"] for item in output.items]


class FakeStructuredTool:
    def __init__(
        self,
        *,
        func,
        name,
        description,
        args_schema,
        infer_schema,
        handle_validation_error=None,
        **_kwargs,
    ) -> None:
        self.func = func
        self.name = name
        self.description = description
        self.args_schema = args_schema
        self.infer_schema = infer_schema
        self.handle_validation_error = handle_validation_error

    @classmethod
    def from_function(cls, **kwargs):
        return cls(**kwargs)

    def invoke(self, arguments: dict[str, Any]) -> str:
        try:
            validated = self.args_schema.model_validate(arguments)
        except ValidationError as exc:
            if self.handle_validation_error is None:
                raise
            return self.handle_validation_error(exc)
        return self.func(**validated.model_dump())


@dataclass
class FakeAIMessage:
    content: Any
    tool_calls: list[dict[str, Any]]
    type: str = "ai"


@dataclass
class FakeToolMessage:
    content: str
    type: str = "tool"


class RecordingGraph:
    def __init__(self, *, tools, response_messages=None, error=None) -> None:
        self.tools = tools
        self.response_messages = response_messages
        self.error = error
        self.invoke_input = None
        self.invoke_config = None
        self.tool_payload = None

    def invoke(self, input, config=None):
        self.invoke_input = input
        self.invoke_config = config
        if self.error is not None:
            raise self.error

        if self.response_messages is not None:
            return {"messages": self.response_messages}

        self.tool_payload = json.loads(
            self.tools[0].invoke({"query": "qdrant"})
        )
        return {
            "messages": [
                {"role": "user", "content": "Qdrant?"},
                FakeAIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "call-1",
                            "name": self.tools[0].name,
                            "args": {"query": "qdrant"},
                        }
                    ],
                ),
                FakeToolMessage(content=json.dumps(self.tool_payload)),
                FakeAIMessage(
                    content="Qdrant evidence answer",
                    tool_calls=[],
                ),
            ]
        }


class RecordingAgentFactory:
    def __init__(self, graph_builder=None) -> None:
        self.kwargs = None
        self.graph = None
        self.graph_builder = graph_builder

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        if self.graph_builder is None:
            self.graph = RecordingGraph(tools=kwargs["tools"])
        else:
            self.graph = self.graph_builder(kwargs["tools"])
        return self.graph




class FakeAgentMiddleware:
    """测试 Runtime Guard 动态 Middleware 的最小父类。"""


@pytest.fixture(autouse=True)
def fake_langchain_core(monkeypatch):
    tools_module = types.ModuleType("langchain_core.tools")
    tools_module.StructuredTool = FakeStructuredTool

    package = types.ModuleType("langchain_core")
    package.tools = tools_module

    middleware_module = types.ModuleType("langchain.agents.middleware")
    middleware_module.AgentMiddleware = FakeAgentMiddleware
    agents_module = types.ModuleType("langchain.agents")
    agents_module.middleware = middleware_module
    langchain_package = types.ModuleType("langchain")
    langchain_package.agents = agents_module

    monkeypatch.setitem(sys.modules, "langchain_core", package)
    monkeypatch.setitem(sys.modules, "langchain_core.tools", tools_module)
    monkeypatch.setitem(sys.modules, "langchain", langchain_package)
    monkeypatch.setitem(sys.modules, "langchain.agents", agents_module)
    monkeypatch.setitem(
        sys.modules,
        "langchain.agents.middleware",
        middleware_module,
    )


def build_context() -> ToolExecutionContext:
    return ToolExecutionContext(
        user_id=7,
        role=UserRole.USER,
        knowledge_base_id=11,
        request_id="req-v21-a2",
        agent_run_id=202,
    )


def test_runner_executes_langchain_candidate_with_same_trusted_tool_boundary():
    tool = DemoTool()
    factory = RecordingAgentFactory()
    runner = LangChainSingleAgentRunner(
        model=object(),
        tools=[tool],
        recursion_limit=9,
        agent_factory=factory,
    )

    result = runner.run(
        db=object(),
        context=build_context(),
        message="  Qdrant?  ",
    )

    assert result.answer == "Qdrant evidence answer"
    assert result.turns == 2
    assert result.tool_call_count == 1
    assert tool.last_context == build_context()
    assert factory.kwargs["system_prompt"] == build_agent_tool_calling_system_prompt()
    assert factory.kwargs["name"] == runner.AGENT_NAME
    assert factory.graph.invoke_input == {
        "messages": [{"role": "user", "content": "Qdrant?"}]
    }
    assert factory.graph.invoke_config == {
        "recursion_limit": 9,
        "max_concurrency": 1,
    }


def test_runner_and_native_llm_share_exact_agent_prompt_version_and_content():
    assert LLMService.AGENT_PROMPT_VERSION == AGENT_TOOL_CALLING_PROMPT_VERSION
    messages = LLMService._build_tool_calling_messages("hello")
    assert messages[0]["content"] == build_agent_tool_calling_system_prompt()


def test_langchain_tool_result_gets_same_citation_reinforcement_as_native_path():
    factory = RecordingAgentFactory()
    runner = LangChainSingleAgentRunner(
        model=object(),
        tools=[KnowledgeTool()],
        agent_factory=factory,
    )

    runner.run(
        db=object(),
        context=build_context(),
        message="Qdrant deployment",
    )

    payload = factory.graph.tool_payload
    assert payload["ok"] is True
    assert payload["evidence_refs"] == ["doc:1:chunk:2"]
    assert payload["_available_source_refs"] == ["doc:1:chunk:2"]
    assert "[source:<source_ref>]" in payload["_agent_citation_instruction"]


def test_runner_extracts_only_text_blocks_from_final_ai_message():
    final_message = FakeAIMessage(
        content=[
            {"type": "reasoning", "text": "hidden reasoning"},
            {"type": "text", "text": "safe answer"},
        ],
        tool_calls=[],
    )
    factory = RecordingAgentFactory(
        graph_builder=lambda tools: RecordingGraph(
            tools=tools,
            response_messages=[
                {"role": "user", "content": "hello"},
                final_message,
            ],
        )
    )
    runner = LangChainSingleAgentRunner(
        model=object(),
        tools=[DemoTool()],
        agent_factory=factory,
    )

    result = runner.run(
        db=object(),
        context=build_context(),
        message="hello",
    )

    assert result.answer == "safe answer"
    assert "hidden reasoning" not in result.answer


def test_runner_maps_graph_recursion_error_to_stable_candidate_error():
    factory = RecordingAgentFactory(
        graph_builder=lambda tools: RecordingGraph(
            tools=tools,
            error=RecursionError("graph recursion limit"),
        )
    )
    runner = LangChainSingleAgentRunner(
        model=object(),
        tools=[DemoTool()],
        agent_factory=factory,
    )

    with pytest.raises(LangChainAgentLimitError) as exc_info:
        runner.run(
            db=object(),
            context=build_context(),
            message="hello",
        )

    assert exc_info.value.code == "langchain_recursion_limit"


def test_runner_rejects_empty_message_and_invalid_graph_state():
    runner = LangChainSingleAgentRunner(
        model=object(),
        tools=[DemoTool()],
        agent_factory=RecordingAgentFactory(),
    )

    with pytest.raises(ValueError, match="message cannot be empty"):
        runner.run(db=object(), context=build_context(), message="   ")

    factory = RecordingAgentFactory(
        graph_builder=lambda tools: RecordingGraph(
            tools=tools,
            response_messages=[],
        )
    )
    invalid_runner = LangChainSingleAgentRunner(
        model=object(),
        tools=[DemoTool()],
        agent_factory=factory,
    )
    with pytest.raises(LangChainAgentError, match="invalid messages state"):
        invalid_runner.run(
            db=object(),
            context=build_context(),
            message="hello",
        )


def test_model_adapter_maps_existing_model_settings_to_chat_openai(monkeypatch):
    captured = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    module = types.ModuleType("langchain_openai")
    module.ChatOpenAI = FakeChatOpenAI
    monkeypatch.setitem(sys.modules, "langchain_openai", module)

    settings = SimpleNamespace(
        model_name="doubao-test",
        model_api_key="secret",
        model_base_url="https://example.test/v1",
    )

    model = LangChainModelAdapter(settings=settings).build()

    assert isinstance(model, FakeChatOpenAI)
    assert captured == {
        "model": "doubao-test",
        "api_key": "secret",
        "base_url": "https://example.test/v1",
        "temperature": 0.2,
        "timeout": 60,
    }


def test_runner_requires_non_empty_tools_and_positive_recursion_limit():
    with pytest.raises(ValueError, match="tools cannot be empty"):
        LangChainSingleAgentRunner(model=object(), tools=[])

    with pytest.raises(ValueError, match="recursion_limit"):
        LangChainSingleAgentRunner(
            model=object(),
            tools=[DemoTool()],
            recursion_limit=0,
        )


def test_runner_injects_observer_middleware_and_reports_final_answer(monkeypatch):
    middleware_module = types.ModuleType("langchain.agents.middleware")

    class FakeAgentMiddleware:
        pass

    middleware_module.AgentMiddleware = FakeAgentMiddleware
    agents_module = types.ModuleType("langchain.agents")
    agents_module.middleware = middleware_module
    package = types.ModuleType("langchain")
    package.agents = agents_module
    monkeypatch.setitem(sys.modules, "langchain", package)
    monkeypatch.setitem(sys.modules, "langchain.agents", agents_module)
    monkeypatch.setitem(
        sys.modules,
        "langchain.agents.middleware",
        middleware_module,
    )

    from app.services.evaluation.agent_observation_collector import (
        AgentEvaluationObservationCollector,
    )

    collector = AgentEvaluationObservationCollector()
    factory = RecordingAgentFactory(
        graph_builder=lambda tools: RecordingGraph(
            tools=tools,
            response_messages=[
                {"role": "user", "content": "Qdrant?"},
                FakeAIMessage(
                    content="Answer [source:doc:1:chunk:2]",
                    tool_calls=[],
                ),
            ],
        )
    )
    runner = LangChainSingleAgentRunner(
        model=object(),
        tools=[DemoTool()],
        agent_factory=factory,
    )

    result = runner.run(
        db=object(),
        context=build_context(),
        message="Qdrant?",
        observer=collector,
    )

    assert result.answer == "Answer [source:doc:1:chunk:2]"
    assert len(factory.kwargs["middleware"]) == 2
    assert collector.build_observed_sources() == ["doc:1:chunk:2"]
