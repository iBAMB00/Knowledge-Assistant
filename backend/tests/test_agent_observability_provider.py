from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.agent.observability import (
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
        self.observation = FakeObservation()
        self.flush_calls = 0
        self.shutdown_calls = 0

    def create_trace_id(self, *, seed: str | None = None) -> str:
        assert seed is not None
        self.trace_id_seeds.append(seed)
        return "a" * 32

    def start_observation(self, **kwargs: Any) -> FakeObservation:
        if self.fail_start:
            raise RuntimeError("provider unavailable")
        self.start_calls.append(kwargs)
        return self.observation

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
