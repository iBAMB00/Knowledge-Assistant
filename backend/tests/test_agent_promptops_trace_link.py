import pytest
from pydantic import ValidationError

from app.agent.observability.noop import NoOpObservabilityProvider
from app.agent.observability.run import start_agent_run_trace
from app.agent.context import ToolExecutionContext
from app.agent.prompt_ops import AgentPromptReference, AgentTracePromptLink
from app.agent.version_snapshot import AgentRuntimeVersionSnapshot
from app.constants.agent_runtime import AgentRuntime
from app.constants.user_role import UserRole


def _context() -> ToolExecutionContext:
    return ToolExecutionContext(
        user_id=7,
        role=UserRole.USER,
        knowledge_base_id=11,
        request_id="req-a6",
        conversation_id=13,
        agent_run_id=17,
    )


def _snapshot() -> AgentRuntimeVersionSnapshot:
    return AgentRuntimeVersionSnapshot(
        agent_version="agent-v2.5",
        prompt_version="1.1.0",
        toolset_version="toolset-v2:test",
        retrieval_config_version="retrieval-v1:test",
    )


def test_trace_session_exposes_safe_prompt_link_without_prompt_content() -> None:
    session = start_agent_run_trace(
        provider=NoOpObservabilityProvider(),
        execution_context=_context(),
        runtime=AgentRuntime.NATIVE,
        version_snapshot=_snapshot(),
        model_provider="test-provider",
        model_name="test-model",
        prompt_id="agent.tool-calling-system",
    )

    link = session.prompt_link
    assert link.trace_id == session.trace_context.trace_id
    assert link.provider_trace_id is None
    assert link.agent_run_id == 17
    assert link.prompt == AgentPromptReference(
        prompt_id="agent.tool-calling-system",
        prompt_version="1.1.0",
    )
    dumped = link.model_dump()
    assert "content" not in dumped["prompt"]
    assert "template" not in dumped["prompt"]


def test_promptops_contract_forbids_raw_prompt_payloads() -> None:
    with pytest.raises(ValidationError):
        AgentTracePromptLink(
            trace_id="trace-1",
            runtime=AgentRuntime.NATIVE,
            agent_version="agent-v2.5",
            prompt=AgentPromptReference(
                prompt_id="agent.tool-calling-system",
                prompt_version="1.1.0",
            ),
            prompt_content="do-not-trace",
        )
