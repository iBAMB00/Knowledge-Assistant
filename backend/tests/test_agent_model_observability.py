from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

import app.services.llm_service as llm_service_module
from app.agent.context import ToolExecutionContext
from app.agent.frameworks.langchain.model_observability import (
    LangChainModelObservabilityBridge,
)
from app.agent.observability import (
    AgentModelCallContext,
    AgentModelCallMode,
    AgentModelTracer,
    AgentModelUsage,
    AgentObservationKind,
    AgentSpanContext,
    AgentTraceContext,
    extract_langchain_usage,
    extract_openai_usage,
    start_agent_run_trace,
)
from app.agent.observability.provider import AgentModelCallHandle
from app.agent.tools.base import ToolContract, ToolRiskLevel
from app.agent.version_snapshot import AgentRuntimeVersionSnapshot
from app.constants.agent_runtime import AgentRuntime
from app.constants.user_role import UserRole
from app.services.llm_service import LLMService


@dataclass
class RecordingModelHandle:
    span_id: str
    finishes: list[dict[str, Any]] = field(default_factory=list)
    fail_finish: bool = False

    def finish(
        self,
        *,
        ok: bool = True,
        usage: AgentModelUsage | None = None,
        error_code: str | None = None,
    ) -> None:
        if self.fail_finish:
            raise RuntimeError("provider-finish-error")
        self.finishes.append(
            {"ok": ok, "usage": usage, "error_code": error_code}
        )


@dataclass
class RecordingTraceHandle:
    trace_id: str
    provider_trace_id: str | None = "provider-trace"
    model_contexts: list[AgentModelCallContext] = field(default_factory=list)
    model_handles: list[RecordingModelHandle] = field(default_factory=list)
    finishes: list[dict[str, Any]] = field(default_factory=list)
    fail_model_start: bool = False
    fail_finish: bool = False

    def start_model_call(
        self,
        *,
        call_context: AgentModelCallContext,
    ) -> AgentModelCallHandle:
        if self.fail_model_start:
            raise RuntimeError("provider-start-error")
        self.model_contexts.append(call_context)
        handle = RecordingModelHandle(span_id=call_context.span.span_id)
        self.model_handles.append(handle)
        return handle

    def finish(
        self,
        *,
        ok: bool = True,
        error_code: str | None = None,
        metrics: Any | None = None,
    ) -> None:
        if self.fail_finish:
            raise RuntimeError("provider-root-finish-error")
        self.finishes.append(
            {"ok": ok, "error_code": error_code, "metrics": metrics}
        )


class RecordingProvider:
    name = "recording"
    enabled = True

    def __init__(self, *, fail_start: bool = False) -> None:
        self.fail_start = fail_start
        self.trace_contexts: list[AgentTraceContext] = []
        self.handles: list[RecordingTraceHandle] = []

    def start_trace(
        self,
        *,
        trace_context: AgentTraceContext,
        name: str = "agent.run",
    ) -> RecordingTraceHandle:
        del name
        if self.fail_start:
            raise RuntimeError("provider-start-error")
        self.trace_contexts.append(trace_context)
        handle = RecordingTraceHandle(trace_id=trace_context.trace_id)
        self.handles.append(handle)
        return handle

    def flush(self) -> None:
        return None

    def shutdown(self) -> None:
        return None


def _trace() -> AgentTraceContext:
    return AgentTraceContext(
        trace_id="1" * 32,
        request_id="req-model-1",
        runtime=AgentRuntime.NATIVE,
        user_id=7,
        knowledge_base_id=11,
        agent_run_id=17,
        agent_version="agent-v2.5",
        prompt_id="agent.tool-calling-system",
        prompt_version="1.1.0",
        toolset_version="toolset-v2:test",
        retrieval_config_version="retrieval-v1:test",
    )


def _model_tracer(handle: RecordingTraceHandle | None = None) -> AgentModelTracer:
    trace = _trace()
    return AgentModelTracer(
        trace_context=trace,
        trace_handle=handle or RecordingTraceHandle(trace_id=trace.trace_id),
        model_provider="openai-compatible",
        model_name="test-model",
        prompt_id="agent.tool-calling-system",
        prompt_version="1.1.0",
    )


def _tool_contract() -> ToolContract:
    return ToolContract(
        name="search_knowledge",
        version="1.0.0",
        description="Search knowledge.",
        risk_level=ToolRiskLevel.READ_ONLY,
        input_schema={"type": "object", "properties": {}},
        output_schema={"type": "object", "properties": {}},
    )


def test_model_call_contract_forbids_prompt_and_output_payloads() -> None:
    span = AgentSpanContext(
        trace_id="trace-1",
        span_id="span-1",
        kind=AgentObservationKind.MODEL,
        name="model.call",
    )

    with pytest.raises(ValidationError):
        AgentModelCallContext(
            span=span,
            model_provider="provider",
            model_name="model",
            prompt_id="prompt",
            prompt_version="1.0.0",
            mode=AgentModelCallMode.TOOL_CALLING,
            prompt="secret prompt",
        )


def test_openai_usage_is_normalized_without_model_payload() -> None:
    response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=120,
            completion_tokens=30,
            total_tokens=150,
        )
    )

    assert extract_openai_usage(response) == AgentModelUsage(
        input_tokens=120,
        output_tokens=30,
        total_tokens=150,
    )


def test_langchain_usage_is_read_from_model_response_message() -> None:
    result = SimpleNamespace(
        result=[
            SimpleNamespace(
                usage_metadata={
                    "input_tokens": 80,
                    "output_tokens": 20,
                    "total_tokens": 100,
                }
            )
        ]
    )

    assert extract_langchain_usage(result) == AgentModelUsage(
        input_tokens=80,
        output_tokens=20,
        total_tokens=100,
    )


def test_model_tracer_is_fail_open_for_provider_start_and_finish() -> None:
    failing_start = RecordingTraceHandle(
        trace_id=_trace().trace_id,
        fail_model_start=True,
    )
    handle = _model_tracer(failing_start).start_call(turn=1)
    handle.finish(ok=True, usage=AgentModelUsage(total_tokens=1))

    failing_finish = RecordingTraceHandle(trace_id=_trace().trace_id)
    traced = _model_tracer(failing_finish).start_call(turn=2)
    failing_finish.model_handles[0].fail_finish = True
    traced.finish(ok=False, error_code="E_MODEL")


def test_langchain_bridge_wraps_actual_handler_and_records_usage() -> None:
    trace_handle = RecordingTraceHandle(trace_id=_trace().trace_id)
    bridge = LangChainModelObservabilityBridge(_model_tracer(trace_handle))
    response = SimpleNamespace(
        result=[
            SimpleNamespace(
                usage_metadata={
                    "input_tokens": 11,
                    "output_tokens": 7,
                    "total_tokens": 18,
                }
            )
        ]
    )

    actual = bridge.observe_model_call(
        request=SimpleNamespace(),
        handler=lambda _request: response,
    )

    assert actual is response
    assert len(trace_handle.model_contexts) == 1
    assert trace_handle.model_contexts[0].turn == 1
    assert trace_handle.model_handles[0].finishes == [
        {
            "ok": True,
            "usage": AgentModelUsage(
                input_tokens=11,
                output_tokens=7,
                total_tokens=18,
            ),
            "error_code": None,
        }
    ]


def test_langchain_bridge_records_safe_error_type_and_reraises() -> None:
    trace_handle = RecordingTraceHandle(trace_id=_trace().trace_id)
    bridge = LangChainModelObservabilityBridge(_model_tracer(trace_handle))

    with pytest.raises(ValueError, match="private detail"):
        bridge.observe_model_call(
            request=SimpleNamespace(),
            handler=lambda _request: (_ for _ in ()).throw(
                ValueError("private detail")
            ),
        )

    assert trace_handle.model_handles[0].finishes == [
        {"ok": False, "usage": None, "error_code": "ValueError"}
    ]


def test_llm_service_records_real_openai_compatible_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="answer", tool_calls=None)
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=90,
            completion_tokens=10,
            total_tokens=100,
        ),
    )

    class FakeCompletions:
        def create(self, **_kwargs: Any) -> Any:
            return response

    class FakeClient:
        chat = SimpleNamespace(completions=FakeCompletions())

    settings = SimpleNamespace(
        model_name="test-model",
        model_api_key="key",
        model_base_url="http://model",
    )
    monkeypatch.setattr(llm_service_module, "get_settings", lambda: settings)
    monkeypatch.setattr(llm_service_module, "OpenAI", lambda **_kwargs: FakeClient())

    service = LLMService()
    trace_handle = RecordingTraceHandle(trace_id=_trace().trace_id)
    result = service.chat_with_tool_history(
        message="question",
        tool_contracts=[_tool_contract()],
        history=(),
        model_tracer=_model_tracer(trace_handle),
        model_turn=3,
    )

    assert result.content == "answer"
    assert trace_handle.model_contexts[0].turn == 3
    assert trace_handle.model_handles[0].finishes == [
        {
            "ok": True,
            "usage": AgentModelUsage(
                input_tokens=90,
                output_tokens=10,
                total_tokens=100,
            ),
            "error_code": None,
        }
    ]


def test_root_trace_session_uses_persisted_agent_run_and_runtime() -> None:
    provider = RecordingProvider()
    context = ToolExecutionContext(
        user_id=7,
        role=UserRole.USER,
        knowledge_base_id=11,
        request_id="req-root-1",
        conversation_id=13,
        agent_run_id=17,
    )
    snapshot = AgentRuntimeVersionSnapshot(
        agent_version="langgraph-v1:1.0",
        prompt_version="1.1.0",
        toolset_version="toolset-v2:test",
        retrieval_config_version="retrieval-v1:test",
    )

    session = start_agent_run_trace(
        provider=provider,
        execution_context=context,
        runtime=AgentRuntime.LANGGRAPH,
        version_snapshot=snapshot,
        model_provider="provider",
        model_name="model",
        prompt_id="agent.tool-calling-system",
        thread_id="conversation:13",
    )

    assert session is not None
    trace = provider.trace_contexts[0]
    assert trace.agent_run_id == 17
    assert trace.runtime is AgentRuntime.LANGGRAPH
    assert trace.thread_id == "conversation:13"
    metrics = session.finish(ok=True)
    assert metrics.success is True
    assert provider.handles[0].finishes == [
        {"ok": True, "error_code": None, "metrics": metrics}
    ]


def test_root_trace_start_is_fail_open() -> None:
    provider = RecordingProvider(fail_start=True)
    context = ToolExecutionContext(
        user_id=7,
        role=UserRole.USER,
        knowledge_base_id=11,
        request_id="req-root-fail",
        agent_run_id=17,
    )
    snapshot = AgentRuntimeVersionSnapshot(
        agent_version="native-v1",
        prompt_version="1.1.0",
        toolset_version="toolset-v2:test",
        retrieval_config_version="retrieval-v1:test",
    )

    session = start_agent_run_trace(
        provider=provider,
        execution_context=context,
        runtime=AgentRuntime.NATIVE,
        version_snapshot=snapshot,
        model_provider="provider",
        model_name="model",
        prompt_id="agent.tool-calling-system",
    )
    assert session.trace_handle.provider_trace_id is None
    assert session.finish(ok=False, error_code="RuntimeError").success is False
