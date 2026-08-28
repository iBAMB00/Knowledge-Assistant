from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from app.agent.observability import (
    AgentComponentResult,
    AgentComponentTracer,
    AgentModelPricing,
    AgentModelTracer,
    AgentErrorStage,
    AgentModelUsage,
    AgentObservationKind,
    AgentRunMetrics,
    AgentRunOutcome,
    build_observation_error,
    AgentRunMetricsCollector,
    AgentTraceContext,
    LangfuseObservabilityProvider,
    NoOpTraceHandle,
    estimate_model_cost_usd,
)
from app.constants.agent_runtime import AgentRuntime


class FakeClock:
    def __init__(self) -> None:
        self.now_ns = 0

    def __call__(self) -> int:
        return self.now_ns

    def advance_ms(self, value: float) -> None:
        self.now_ns += int(value * 1_000_000)


def _pricing() -> AgentModelPricing:
    return AgentModelPricing(
        provider="test-provider",
        model_name="test-model",
        input_usd_per_million_tokens=Decimal("2"),
        output_usd_per_million_tokens=Decimal("8"),
        version="pricing-2026-08",
    )


def _trace() -> AgentTraceContext:
    return AgentTraceContext(
        trace_id="1" * 32,
        request_id="req-a5",
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


def test_model_pricing_estimate_uses_explicit_snapshot() -> None:
    cost = estimate_model_cost_usd(
        pricing=_pricing(),
        usage=AgentModelUsage(
            input_tokens=1000,
            output_tokens=500,
            total_tokens=1500,
        ),
    )

    assert cost == Decimal("0.006")


def test_run_metrics_aggregate_tokens_calls_latency_and_cost() -> None:
    clock = FakeClock()
    collector = AgentRunMetricsCollector(pricing=_pricing(), clock_ns=clock)

    model_started = collector.begin_call()
    clock.advance_ms(120)
    collector.record_model(
        started_ns=model_started,
        ok=True,
        usage=AgentModelUsage(
            input_tokens=1000,
            output_tokens=500,
            total_tokens=1500,
        ),
    )

    tool_started = collector.begin_call()
    clock.advance_ms(40)
    collector.record_component(
        kind=AgentObservationKind.TOOL,
        started_ns=tool_started,
        ok=True,
    )

    retrieval_started = collector.begin_call()
    clock.advance_ms(30)
    collector.record_component(
        kind=AgentObservationKind.RETRIEVAL,
        started_ns=retrieval_started,
        ok=True,
    )

    mcp_started = collector.begin_call()
    clock.advance_ms(10)
    collector.record_component(
        kind=AgentObservationKind.MCP,
        started_ns=mcp_started,
        ok=False,
    )

    graph_started = collector.begin_call()
    clock.advance_ms(5)
    collector.record_component(
        kind=AgentObservationKind.GRAPH_NODE,
        started_ns=graph_started,
        ok=True,
    )

    metrics = collector.finish(success=False, error_type="ToolExecutionError")

    assert metrics == AgentRunMetrics(
        success=False,
        error_type="ToolExecutionError",
        run_latency_ms=205.0,
        model_calls=1,
        tool_calls=1,
        retrieval_calls=1,
        mcp_calls=1,
        graph_node_calls=1,
        failed_calls=1,
        input_tokens=1000,
        output_tokens=500,
        total_tokens=1500,
        model_usage_missing_calls=0,
        model_latency_ms=120.0,
        tool_latency_ms=40.0,
        retrieval_latency_ms=30.0,
        mcp_latency_ms=10.0,
        graph_node_latency_ms=5.0,
        estimated_cost_usd=Decimal("0.00600000"),
        pricing_configured=True,
        cost_estimate_complete=True,
        pricing_version="pricing-2026-08",
    )
    assert collector.finish(success=True) is metrics


def test_metrics_still_work_when_external_provider_is_noop() -> None:
    clock = FakeClock()
    collector = AgentRunMetricsCollector(pricing=None, clock_ns=clock)
    trace = _trace()
    trace_handle = NoOpTraceHandle(trace_id=trace.trace_id)
    model_tracer = AgentModelTracer(
        trace_context=trace,
        trace_handle=trace_handle,
        model_provider="test-provider",
        model_name="test-model",
        prompt_id="agent.tool-calling-system",
        prompt_version="1.1.0",
        metrics_collector=collector,
    )
    component_tracer = AgentComponentTracer(
        trace_context=trace,
        trace_handle=trace_handle,
        metrics_collector=collector,
    )

    model = model_tracer.start_call(turn=1)
    clock.advance_ms(15)
    model.finish(
        usage=AgentModelUsage(
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
        )
    )

    tool = component_tracer.start_tool(
        tool_name="search_knowledge",
        tool_version="1.0.0",
        tool_source="local",
        call_id="call-1",
        turn=1,
    )
    clock.advance_ms(8)
    retrieval = component_tracer.start_retrieval(
        parent_span_id=tool.span_id,
        top_k=5,
        turn=1,
    )
    clock.advance_ms(4)
    retrieval.finish(result=AgentComponentResult(result_count=2, evidence_count=2))
    tool.finish(result=AgentComponentResult(result_count=2, evidence_count=2))

    metrics = collector.finish(success=True)

    assert metrics.model_calls == 1
    assert metrics.tool_calls == 1
    assert metrics.retrieval_calls == 1
    assert metrics.total_tokens == 15
    assert metrics.estimated_cost_usd is None
    assert metrics.pricing_configured is False
    assert metrics.cost_estimate_complete is False


@dataclass
class FakeObservation:
    id: str
    trace_id: str = "provider-trace"
    updates: list[dict[str, Any]] = field(default_factory=list)
    ended: int = 0

    def update(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)

    def end(self) -> None:
        self.ended += 1


class FakeLangfuseClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.observations: list[FakeObservation] = []

    def create_trace_id(self, *, seed: str | None = None) -> str:
        del seed
        return "a" * 32

    def start_observation(self, **kwargs: Any) -> FakeObservation:
        self.calls.append(kwargs)
        observation = FakeObservation(id=f"obs-{len(self.calls)}")
        self.observations.append(observation)
        return observation

    def flush(self) -> None:
        return None

    def shutdown(self) -> None:
        return None


def test_langfuse_root_receives_safe_run_metrics_summary() -> None:
    client = FakeLangfuseClient()
    provider = LangfuseObservabilityProvider(client=client)
    handle = provider.start_trace(trace_context=_trace())
    metrics = AgentRunMetrics(
        success=True,
        run_latency_ms=321.5,
        model_calls=2,
        tool_calls=1,
        retrieval_calls=1,
        input_tokens=200,
        output_tokens=80,
        total_tokens=280,
        model_latency_ms=250.0,
        tool_latency_ms=40.0,
        retrieval_latency_ms=30.0,
        estimated_cost_usd=Decimal("0.00123456"),
        pricing_configured=True,
        cost_estimate_complete=True,
        pricing_version="pricing-2026-08",
    )

    handle.finish(metrics=metrics)

    update = client.observations[0].updates[0]
    metadata = update["metadata"]
    assert metadata["runtime"] == "native"
    assert metadata["request_id"] == "req-a5"
    assert metadata["run_latency_ms"] == 321.5
    assert metadata["model_calls"] == 2
    assert metadata["tool_calls"] == 1
    assert metadata["total_tokens"] == 280
    assert metadata["estimated_cost_usd"] == 0.00123456
    assert metadata["cost_estimate_complete"] is True
    assert metadata["pricing_version"] == "pricing-2026-08"
    serialized = repr(metadata)
    assert "prompt_content" not in serialized
    assert "input" not in metadata
    assert "output" not in metadata
    assert "tool_arguments" not in serialized
    assert "retrieved_document" not in serialized



def test_successful_run_with_failed_child_is_degraded_and_preserves_first_error() -> None:
    clock = FakeClock()
    collector = AgentRunMetricsCollector(clock_ns=clock)
    started = collector.begin_call()
    clock.advance_ms(5)
    error = build_observation_error(
        RuntimeError("reranker request failed: HTTP 403: token=private"),
        stage=AgentErrorStage.RETRIEVAL,
        error_code="execution_failed",
    )
    collector.record_component(
        kind=AgentObservationKind.RETRIEVAL,
        started_ns=started,
        ok=False,
        error=error,
    )

    metrics = collector.finish(success=True)

    assert metrics.outcome is AgentRunOutcome.DEGRADED
    assert metrics.first_error is not None
    assert metrics.first_error.error_code == "execution_failed"
    assert metrics.first_error.http_status == 403
    assert "private" not in (metrics.first_error.safe_message or "")


def test_warning_component_degrades_successful_run_without_counting_failure() -> None:
    clock = FakeClock()
    collector = AgentRunMetricsCollector(clock_ns=clock)
    started = collector.begin_call()
    warning = build_observation_error(
        RuntimeError("reranker unavailable"),
        stage=AgentErrorStage.RETRIEVAL,
        error_code="reranker_execution_failed",
        fail_open=True,
    )
    clock.advance_ms(3)
    collector.record_component(
        kind=AgentObservationKind.RETRIEVAL,
        started_ns=started,
        ok=True,
        warning=warning,
    )

    metrics = collector.finish(success=True)

    assert metrics.outcome is AgentRunOutcome.DEGRADED
    assert metrics.failed_calls == 0
    assert metrics.warning_calls == 1
    assert metrics.first_warning is not None
    assert metrics.first_warning.fail_open is True


def test_explicit_waiting_and_cancelled_outcomes_override_failed_default() -> None:
    waiting = AgentRunMetricsCollector(clock_ns=FakeClock()).finish(
        success=False,
        error_type="approval_required",
        outcome=AgentRunOutcome.WAITING,
    )
    cancelled = AgentRunMetricsCollector(clock_ns=FakeClock()).finish(
        success=False,
        error_type="agent_cancelled",
        outcome=AgentRunOutcome.CANCELLED,
    )

    assert waiting.outcome is AgentRunOutcome.WAITING
    assert cancelled.outcome is AgentRunOutcome.CANCELLED


def test_model_tracer_forwards_explicit_pricing_as_langfuse_cost_details() -> None:
    clock = FakeClock()
    collector = AgentRunMetricsCollector(pricing=_pricing(), clock_ns=clock)
    client = FakeLangfuseClient()
    provider = LangfuseObservabilityProvider(client=client)
    trace = _trace()
    root = provider.start_trace(trace_context=trace)
    tracer = AgentModelTracer(
        trace_context=trace,
        trace_handle=root,
        model_provider="test-provider",
        model_name="test-model",
        prompt_id="agent.tool-calling-system",
        prompt_version="1.1.0",
        metrics_collector=collector,
    )

    handle = tracer.start_call(turn=1)
    clock.advance_ms(5)
    handle.finish(
        usage=AgentModelUsage(
            input_tokens=1000,
            output_tokens=500,
            total_tokens=1500,
        )
    )

    update = client.observations[1].updates[0]
    assert update["usage_details"] == {"input": 1000, "output": 500, "total": 1500}
    assert update["cost_details"] == {
        "input": 0.002,
        "output": 0.004,
        "total": 0.006,
    }
