from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.agent.observability import (
    AgentModelCallContext,
    AgentModelCallMode,
    AgentModelUsage,
    AgentObservationKind,
    AgentSpanContext,
    AgentTraceContext,
    LangfuseObservabilityProvider,
    NoOpObservabilityProvider,
    build_observability_provider,
)
from app.constants.agent_runtime import AgentRuntime


@dataclass
class FakeObservation:
    id: str = "obs-1"
    trace_id: str = "provider-trace"
    updates: list[dict[str, Any]] = field(default_factory=list)
    end_calls: int = 0

    def update(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)

    def end(self) -> None:
        self.end_calls += 1


class FakeLangfuseClient:
    def __init__(self, *, fail_start: bool = False) -> None:
        self.fail_start = fail_start
        self.start_calls: list[dict[str, Any]] = []
        self.trace_id_seeds: list[str] = []
        self.observations: list[FakeObservation] = []
        self.flush_calls = 0
        self.shutdown_calls = 0
        self.score_calls: list[dict[str, Any]] = []

    def create_trace_id(self, *, seed: str | None = None) -> str:
        assert seed is not None
        self.trace_id_seeds.append(seed)
        return "a" * 32

    @property
    def observation(self) -> FakeObservation:
        return self.observations[0]

    def start_observation(self, **kwargs: Any) -> FakeObservation:
        if self.fail_start:
            raise RuntimeError("provider unavailable")
        self.start_calls.append(kwargs)
        observation = FakeObservation(id=f"obs-{len(self.observations) + 1}")
        self.observations.append(observation)
        return observation

    def create_score(self, **kwargs: Any) -> None:
        self.score_calls.append(kwargs)

    def flush(self) -> None:
        self.flush_calls += 1

    def shutdown(self) -> None:
        self.shutdown_calls += 1


class FakeSettings:
    langfuse_enabled = False


def _trace(trace_id: str = "1" * 32) -> AgentTraceContext:
    return AgentTraceContext(
        trace_id=trace_id,
        request_id="req-1",
        runtime=AgentRuntime.LANGGRAPH,
        user_id=7,
        knowledge_base_id=11,
        conversation_id=13,
        thread_id="thread-1",
        agent_run_id=17,
        agent_version="agent-v2.5",
        prompt_id="agent.tool-calling-system",
        prompt_version="1.1.0",
        toolset_version="toolset-v2:abc",
        retrieval_config_version="retrieval-v1:def",
    )


def test_disabled_observability_builds_noop_provider() -> None:
    provider = build_observability_provider(FakeSettings())  # type: ignore[arg-type]

    assert isinstance(provider, NoOpObservabilityProvider)
    assert provider.enabled is False
    handle = provider.start_trace(trace_context=_trace())
    assert handle.trace_id == "1" * 32
    assert handle.provider_trace_id is None


def test_factory_fails_open_when_langfuse_initialization_fails() -> None:
    class EnabledSettings(FakeSettings):
        langfuse_enabled = True

    def broken_builder(_settings: Any) -> Any:
        raise RuntimeError("bad credentials")

    provider = build_observability_provider(
        EnabledSettings(),  # type: ignore[arg-type]
        langfuse_builder=broken_builder,
    )

    assert isinstance(provider, NoOpObservabilityProvider)


def test_langfuse_provider_starts_agent_trace_with_safe_metadata_only() -> None:
    client = FakeLangfuseClient()
    provider = LangfuseObservabilityProvider(client=client)

    handle = provider.start_trace(trace_context=_trace(), name="agent.request")

    assert handle.provider_trace_id == "1" * 32
    call = client.start_calls[0]
    assert call["trace_context"] == {"trace_id": "1" * 32}
    assert call["as_type"] == "agent"
    assert call["name"] == "agent.request"
    assert call["version"] == "agent-v2.5"
    assert call["metadata"] == {
        "internal_trace_id": "1" * 32,
        "request_id": "req-1",
        "runtime": "langgraph",
        "user_id": 7,
        "knowledge_base_id": 11,
        "agent_version": "agent-v2.5",
        "prompt_id": "agent.tool-calling-system",
        "prompt_version": "1.1.0",
        "toolset_version": "toolset-v2:abc",
        "retrieval_config_version": "retrieval-v1:def",
        "conversation_id": 13,
        "thread_id": "thread-1",
        "agent_run_id": "17",
    }
    assert "prompt" not in call
    assert "input" not in call
    assert "output" not in call


def test_non_w3c_internal_trace_id_is_deterministically_mapped() -> None:
    client = FakeLangfuseClient()
    provider = LangfuseObservabilityProvider(client=client)

    handle = provider.start_trace(trace_context=_trace("trace-readable"))

    assert handle.trace_id == "trace-readable"
    assert handle.provider_trace_id == "a" * 32
    assert client.trace_id_seeds == ["trace-readable"]


def test_langfuse_trace_lifecycle_is_idempotent_and_bounded() -> None:
    client = FakeLangfuseClient()
    provider = LangfuseObservabilityProvider(client=client)
    handle = provider.start_trace(trace_context=_trace())

    handle.finish(ok=False, error_code=" E_TIMEOUT ")
    handle.finish(ok=True)
    provider.flush()
    provider.shutdown()

    assert client.observation.updates == [
        {"level": "ERROR", "status_message": "E_TIMEOUT"}
    ]
    assert client.observation.end_calls == 1
    assert client.flush_calls == 1
    assert client.shutdown_calls == 1


def test_langfuse_start_failure_degrades_to_noop_handle() -> None:
    client = FakeLangfuseClient(fail_start=True)
    provider = LangfuseObservabilityProvider(client=client)

    handle = provider.start_trace(trace_context=_trace())

    assert handle.trace_id == "1" * 32
    assert handle.provider_trace_id is None
    handle.finish(ok=False, error_code="E_PROVIDER")


def test_langfuse_model_generation_is_child_of_agent_trace_without_payloads() -> None:
    client = FakeLangfuseClient()
    provider = LangfuseObservabilityProvider(client=client)
    trace = _trace()
    trace_handle = provider.start_trace(trace_context=trace)
    call_context = AgentModelCallContext(
        span=AgentSpanContext(
            trace_id=trace.trace_id,
            span_id="model-span-1",
            kind=AgentObservationKind.MODEL,
            name="model.call",
        ),
        model_provider="openai-compatible",
        model_name="test-model",
        prompt_id="agent.tool-calling-system",
        prompt_version="1.1.0",
        mode=AgentModelCallMode.TOOL_CALLING,
        turn=2,
    )

    model_handle = trace_handle.start_model_call(call_context=call_context)
    model_handle.finish(
        usage=AgentModelUsage(
            input_tokens=120,
            output_tokens=30,
            total_tokens=150,
        )
    )

    assert len(client.start_calls) == 2
    root_call, model_call = client.start_calls
    assert root_call["as_type"] == "agent"
    assert model_call["as_type"] == "generation"
    assert model_call["trace_context"] == {
        "trace_id": "1" * 32,
        "parent_span_id": "obs-1",
    }
    assert model_call["model"] == "test-model"
    assert model_call["version"] == "1.1.0"
    assert model_call["metadata"] == {
        "internal_span_id": "model-span-1",
        "model_provider": "openai-compatible",
        "prompt_id": "agent.tool-calling-system",
        "prompt_version": "1.1.0",
        "mode": "tool_calling",
        "turn": 2,
    }
    assert "input" not in model_call
    assert "output" not in model_call
    assert client.observations[1].updates == [
        {"usage_details": {"input": 120, "output": 30, "total": 150}}
    ]
    assert client.observations[1].end_calls == 1


class RecordingCorrelationScope:
    def __init__(self, sink: list[dict[str, str]], attrs: dict[str, str]) -> None:
        self.sink = sink
        self.attrs = attrs
        self.exit_calls = 0

    def __enter__(self):
        self.sink.append(self.attrs)
        return self

    def __exit__(self, exc_type, exc, tb):
        self.exit_calls += 1
        return False


def test_langfuse_v4_correlation_maps_user_and_conversation_to_native_attributes() -> None:
    client = FakeLangfuseClient()
    correlation_calls: list[dict[str, str]] = []
    scopes: list[RecordingCorrelationScope] = []

    def factory(**attrs: str):
        scope = RecordingCorrelationScope(correlation_calls, attrs)
        scopes.append(scope)
        return scope

    provider = LangfuseObservabilityProvider(
        client=client,
        correlation_scope_factory=factory,
    )
    handle = provider.start_trace(trace_context=_trace(), name="agent.chat")

    expected = {
        "trace_name": "agent.chat",
        "user_id": "7",
        "session_id": "conversation:13",
    }
    assert correlation_calls == [expected]
    # Correlation scope is deliberately bounded to observation creation so it
    # cannot leak across SSE/generator yield boundaries.
    assert scopes[0].exit_calls == 1

    model_context = AgentModelCallContext(
        span=AgentSpanContext(
            trace_id=handle.trace_id,
            span_id="model-correlated",
            kind=AgentObservationKind.MODEL,
            name="model.call",
        ),
        model_provider="test",
        model_name="test-model",
        prompt_id="agent.tool-calling-system",
        prompt_version="1.1.0",
        mode=AgentModelCallMode.TOOL_CALLING,
    )
    handle.start_model_call(call_context=model_context)
    assert correlation_calls == [expected, expected]
    assert scopes[1].exit_calls == 1
    handle.finish()


def test_langfuse_input_preview_is_opt_in_and_bounded() -> None:
    trace = _trace().model_copy(update={"input_preview": "敏感测试问题" * 20})
    client = FakeLangfuseClient()
    disabled = LangfuseObservabilityProvider(client=client)
    disabled.start_trace(trace_context=trace)
    assert "input" not in client.start_calls[0]

    client2 = FakeLangfuseClient()
    enabled = LangfuseObservabilityProvider(
        client=client2,
        capture_input_preview=True,
        input_preview_max_chars=30,
    )
    enabled.start_trace(trace_context=trace)
    preview = client2.start_calls[0]["input"]["question_preview"]
    assert len(preview) <= 30
    assert preview.endswith("…")


def test_langfuse_trace_score_publish_is_fail_open_and_uses_provider_trace_id() -> None:
    client = FakeLangfuseClient()
    provider = LangfuseObservabilityProvider(client=client)

    assert provider.publish_trace_score(
        provider_trace_id="1" * 32,
        name="eval.task_success",
        value=1.0,
        data_type="BOOLEAN",
        comment="case=test",
    ) is True
    assert client.score_calls == [
        {
            "trace_id": "1" * 32,
            "name": "eval.task_success",
            "value": 1.0,
            "data_type": "BOOLEAN",
            "comment": "case=test",
        }
    ]


def test_langfuse_generation_accepts_explicit_cost_details() -> None:
    from decimal import Decimal
    from app.agent.observability.contracts import AgentModelCost

    client = FakeLangfuseClient()
    provider = LangfuseObservabilityProvider(client=client)
    trace = _trace()
    trace_handle = provider.start_trace(trace_context=trace)
    call_context = AgentModelCallContext(
        span=AgentSpanContext(
            trace_id=trace.trace_id,
            span_id="model-cost",
            kind=AgentObservationKind.MODEL,
            name="model.call",
        ),
        model_provider="test-provider",
        model_name="test-model",
        prompt_id="agent.tool-calling-system",
        prompt_version="1.1.0",
        mode=AgentModelCallMode.TOOL_CALLING,
    )

    handle = trace_handle.start_model_call(call_context=call_context)
    handle.finish(
        usage=AgentModelUsage(input_tokens=100, output_tokens=20, total_tokens=120),
        cost=AgentModelCost(
            input_cost_usd=Decimal("0.001"),
            output_cost_usd=Decimal("0.002"),
            total_cost_usd=Decimal("0.003"),
        ),
    )

    assert client.observations[1].updates == [
        {
            "usage_details": {"input": 100, "output": 20, "total": 120},
            "cost_details": {"input": 0.001, "output": 0.002, "total": 0.003},
        }
    ]


def test_langfuse_root_waiting_and_cancelled_are_warnings_not_errors() -> None:
    from app.agent.observability.contracts import AgentRunMetrics, AgentRunOutcome

    for outcome, code in (
        (AgentRunOutcome.WAITING, "approval_required"),
        (AgentRunOutcome.CANCELLED, "agent_cancelled"),
    ):
        client = FakeLangfuseClient()
        provider = LangfuseObservabilityProvider(client=client)
        handle = provider.start_trace(trace_context=_trace())
        metrics = AgentRunMetrics(
            success=False,
            outcome=outcome,
            error_type=code,
            run_latency_ms=10,
        )
        handle.finish(ok=False, error_code=code, metrics=metrics)
        update = client.observation.updates[0]
        assert update["level"] == "WARNING"
        assert update["status_message"] == code


def test_langfuse_score_publish_forwards_idempotent_score_id() -> None:
    client = FakeLangfuseClient()
    provider = LangfuseObservabilityProvider(client=client)

    assert provider.publish_trace_score(
        provider_trace_id="1" * 32,
        name="eval.task_success",
        value=1.0,
        data_type="BOOLEAN",
        score_id="score-123",
    ) is True
    assert client.score_calls[0]["score_id"] == "score-123"
