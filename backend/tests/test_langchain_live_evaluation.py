import json
import sys
import types
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from app.agent.context import ToolExecutionContext
from app.agent.frameworks.langchain.run_observer_bridge import (
    LangChainRunObserverBridge,
)
from app.agent.model_response import LLMToolCall
from app.constants.user_role import UserRole
from app.schemas.agent_evaluation import (
    AgentEvaluationCase,
    AgentEvaluationCaseCategory,
    AgentEvaluationDataset,
    AgentEvaluationDatasetReference,
    AgentExpectedToolCall,
)
from app.services.evaluation.agent_evaluator import AgentEvaluator
from app.services.evaluation.agent_observation_collector import (
    AgentEvaluationObservationCollector,
)
from app.services.evaluation.langchain_live_evaluation_runner import (
    LangChainLiveEvaluationRunner,
)


class FakeAgentMiddleware:
    """测试 Observer Bridge 动态 Middleware 的最小父类。"""


@dataclass
class FakeToolMessage:
    content: str


@dataclass
class FakeToolCallRequest:
    tool_call: dict[str, Any]


@pytest.fixture
def fake_langchain_middleware(monkeypatch):
    middleware_module = types.ModuleType("langchain.agents.middleware")
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


def test_observer_bridge_maps_langchain_model_and_tool_events_to_existing_collector(
    fake_langchain_middleware,
) -> None:
    collector = AgentEvaluationObservationCollector()
    middleware = LangChainRunObserverBridge(collector).build_middleware()

    middleware.after_model(
        {
            "messages": [
                SimpleNamespace(
                    tool_calls=[
                        {
                            "id": "call-1",
                            "name": "search_knowledge",
                            "args": {"query": "Qdrant", "top_k": 10},
                        }
                    ]
                )
            ]
        },
        runtime=None,
    )

    request = FakeToolCallRequest(
        tool_call={
            "id": "call-1",
            "name": "search_knowledge",
            "args": {"query": "Qdrant", "top_k": 10},
        }
    )
    result = middleware.wrap_tool_call(
        request,
        lambda _request: FakeToolMessage(
            content=json.dumps(
                {
                    "ok": True,
                    "result": {"result_count": 1},
                    "evidence_refs": [
                        "doc:1:chunk:2",
                        "doc:1:chunk:2",
                    ],
                }
            )
        ),
    )
    collector.on_final_answer("Qdrant uses port 6333. [source:doc:1:chunk:2]")

    assert isinstance(result, FakeToolMessage)
    assert collector.build_tool_calls()[0].model_dump() == {
        "tool_name": "search_knowledge",
        "arguments": {"query": "Qdrant", "top_k": 10},
        "error_code": None,
    }
    assert collector.build_retrieved_sources() == ["doc:1:chunk:2"]
    assert collector.build_observed_sources() == ["doc:1:chunk:2"]


def test_observer_bridge_preserves_safe_tool_error_code(
    fake_langchain_middleware,
) -> None:
    collector = AgentEvaluationObservationCollector()
    middleware = LangChainRunObserverBridge(collector).build_middleware()
    middleware.after_model(
        {
            "messages": [
                SimpleNamespace(
                    tool_calls=[
                        {
                            "id": "call-error",
                            "name": "get_document",
                            "args": {"document_id": 999},
                        }
                    ]
                )
            ]
        },
        runtime=None,
    )

    middleware.wrap_tool_call(
        FakeToolCallRequest(
            tool_call={
                "id": "call-error",
                "name": "get_document",
                "args": {"document_id": 999},
            }
        ),
        lambda _request: FakeToolMessage(
            content=json.dumps(
                {
                    "ok": False,
                    "error": {
                        "code": "resource_not_found",
                        "message": "resource not found",
                    },
                }
            )
        ),
    )

    call = collector.build_tool_calls()[0]
    assert call.error_code == "resource_not_found"
    assert collector.build_retrieved_sources() == []


def test_observer_bridge_marks_framework_tool_exception_and_re_raises(
    fake_langchain_middleware,
) -> None:
    collector = AgentEvaluationObservationCollector()
    middleware = LangChainRunObserverBridge(collector).build_middleware()
    middleware.after_model(
        {
            "messages": [
                SimpleNamespace(
                    tool_calls=[
                        {
                            "id": "call-crash",
                            "name": "search_knowledge",
                            "args": {"query": "boom"},
                        }
                    ]
                )
            ]
        },
        runtime=None,
    )

    def raise_error(_request):
        raise RuntimeError("framework crash")

    with pytest.raises(RuntimeError, match="framework crash"):
        middleware.wrap_tool_call(
            FakeToolCallRequest(
                tool_call={
                    "id": "call-crash",
                    "name": "search_knowledge",
                    "args": {"query": "boom"},
                }
            ),
            raise_error,
        )

    assert collector.build_tool_calls()[0].error_code == "framework_tool_error"


class FakeAccessPolicy:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def get_accessible_knowledge_base(
        self,
        *,
        db,
        knowledge_base_id: int,
        user,
    ):
        self.calls.append((knowledge_base_id, user.id))
        return object()


class ScriptedLangChainRunner:
    def run(self, *, db, context, message: str, observer=None):
        if message == "你能做什么？":
            answer = "我可以帮助查询当前知识库。"
            if observer is not None:
                observer.on_final_answer(answer)
            return SimpleNamespace(answer=answer, turns=1, tool_call_count=0)

        call = LLMToolCall(
            id="call-search",
            name="search_knowledge",
            arguments_json='{"query":"Qdrant 部署","top_k":10}',
        )
        if observer is not None:
            observer.on_tool_call_requested(call)
            observer.on_tool_result(
                call_id=call.id,
                tool_name=call.name,
                ok=True,
                error_code=None,
                evidence_refs=["doc:1:chunk:2"],
            )
        answer = "Qdrant uses port 6333. [source:doc:1:chunk:2]"
        if observer is not None:
            observer.on_final_answer(answer)
        return SimpleNamespace(answer=answer, turns=2, tool_call_count=1)


def _context() -> ToolExecutionContext:
    return ToolExecutionContext(
        user_id=7,
        role=UserRole.USER,
        knowledge_base_id=11,
        request_id="langchain-eval-bootstrap",
    )


def test_langchain_live_eval_reuses_existing_observation_and_evaluator_contract() -> None:
    dataset = AgentEvaluationDataset(
        schema_version="1.0",
        dataset_id="langchain-eval",
        dataset_version="1.5.0",
        description="LangChain candidate eval",
        cases=[
            AgentEvaluationCase(
                case_id="direct",
                query="你能做什么？",
                category=AgentEvaluationCaseCategory.NO_TOOL,
                expected_behavior="直接回答",
                allowed_tools=[],
                forbidden_tools=["search_knowledge"],
                expected_answerable=True,
                expected_tool_calls=[],
            ),
            AgentEvaluationCase(
                case_id="search",
                query="Qdrant 部署",
                category=AgentEvaluationCaseCategory.ONE_TOOL,
                expected_behavior="检索并引用",
                allowed_tools=["search_knowledge"],
                forbidden_tools=[],
                expected_answerable=True,
                require_retrieved_evidence=True,
                require_citation=True,
                expected_tool_calls=[
                    AgentExpectedToolCall(tool_name="search_knowledge")
                ],
            ),
        ],
    )
    access_policy = FakeAccessPolicy()

    observations = LangChainLiveEvaluationRunner(
        agent_runner=ScriptedLangChainRunner(),
        access_policy=access_policy,
    ).run_dataset(
        db=object(),
        dataset=dataset,
        context=_context(),
    )

    assert observations.runner_version == "langchain-v1:1.4.0"
    assert access_policy.calls == [(11, 7)]
    assert observations.observations[0].tool_calls == []

    search = observations.observations[1]
    assert search.run_succeeded is True
    assert search.tool_calls[0].tool_name == "search_knowledge"
    assert search.tool_calls[0].arguments == {
        "query": "Qdrant 部署",
        "top_k": 10,
    }
    assert search.retrieved_sources == ["doc:1:chunk:2"]
    assert search.observed_sources == ["doc:1:chunk:2"]

    report = AgentEvaluator().evaluate(
        dataset=dataset,
        dataset_reference=AgentEvaluationDatasetReference(
            schema_version="1.0",
            dataset_id=dataset.dataset_id,
            dataset_version=dataset.dataset_version,
            source_path="/tmp/langchain-eval.json",
            source_sha256="a" * 64,
            total_cases=2,
        ),
        observations=observations,
    )

    assert report.summary.task_success_rate == 1.0
    assert report.summary.tool_selection_accuracy == 1.0
    assert report.summary.tool_execution_accuracy == 1.0
    assert report.summary.required_evidence_success_rate == 1.0
    assert report.summary.required_citation_success_rate == 1.0
    assert report.summary.citation_correctness == 1.0


class CapturingLangChainExecutionService:
    def __init__(self, agent_runner) -> None:
        self.agent_runner = agent_runner
        self.evaluation_versions = []

    def run(
        self,
        *,
        db,
        context,
        message: str,
        observer=None,
        evaluation_version=None,
    ):
        self.evaluation_versions.append(evaluation_version)
        return self.agent_runner.run(
            db=db,
            context=context,
            message=message,
            observer=observer,
        )


def test_langchain_live_eval_forwards_dataset_and_evaluator_versions_to_lifecycle() -> None:
    dataset = AgentEvaluationDataset(
        schema_version="1.0",
        dataset_id="langchain-lifecycle-eval",
        dataset_version="1.5.0",
        description="Lifecycle version forwarding",
        cases=[
            AgentEvaluationCase(
                case_id="direct",
                query="你能做什么？",
                category=AgentEvaluationCaseCategory.NO_TOOL,
                expected_behavior="直接回答",
                allowed_tools=[],
                forbidden_tools=["search_knowledge"],
                expected_answerable=True,
                expected_tool_calls=[],
            )
        ],
    )
    scripted_runner = ScriptedLangChainRunner()
    execution_service = CapturingLangChainExecutionService(scripted_runner)

    observations = LangChainLiveEvaluationRunner(
        agent_runner=scripted_runner,
        access_policy=FakeAccessPolicy(),
        execution_service=execution_service,  # type: ignore[arg-type]
        evaluator_version="1.4.0",
    ).run_dataset(
        db=object(),
        dataset=dataset,
        context=_context(),
    )

    assert observations.observations[0].run_succeeded is True
    assert len(execution_service.evaluation_versions) == 1
    version = execution_service.evaluation_versions[0]
    assert version is not None
    assert version.dataset_version == "1.5.0"
    assert version.evaluator_version == "1.4.0"


class RecordingExecutionObserver:
    def __init__(self) -> None:
        self.started = []
        self.finished = []

    def on_tool_execution_started(self, *, call_id: str, tool_name: str) -> None:
        self.started.append((call_id, tool_name))

    def on_tool_execution_finished(
        self,
        *,
        call_id: str,
        tool_name: str,
        ok: bool,
        error_code: str | None,
        duration_ms: int,
    ) -> None:
        self.finished.append(
            (call_id, tool_name, ok, error_code, duration_ms)
        )


def test_observer_bridge_separates_actual_tool_execution_observer(
    fake_langchain_middleware,
) -> None:
    execution_observer = RecordingExecutionObserver()
    middleware = LangChainRunObserverBridge(
        execution_observer=execution_observer
    ).build_middleware()

    result = middleware.wrap_tool_call(
        FakeToolCallRequest(
            tool_call={
                "id": "call-executed",
                "name": "get_document",
                "args": {"document_id": 1},
            }
        ),
        lambda _request: FakeToolMessage(
            content=json.dumps({"ok": True, "result": {"id": 1}})
        ),
    )

    assert isinstance(result, FakeToolMessage)
    assert execution_observer.started == [
        ("call-executed", "get_document")
    ]
    assert len(execution_observer.finished) == 1
    call_id, tool_name, ok, error_code, duration_ms = (
        execution_observer.finished[0]
    )
    assert (call_id, tool_name, ok, error_code) == (
        "call-executed",
        "get_document",
        True,
        None,
    )
    assert duration_ms >= 0
