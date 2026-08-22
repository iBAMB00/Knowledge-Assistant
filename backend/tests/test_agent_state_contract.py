import pytest
from pydantic import ValidationError

from app.agent.state import (
    AGENT_STATE_SCHEMA_VERSION,
    AgentState,
    AgentThreadIdentity,
)
from app.constants.agent_state_status import AgentStateStatus
from app.constants.conversation_message_role import ConversationMessageRole
from app.constants.conversation_mode import ConversationMode
from app.schemas.conversation_contract import (
    ConversationMessagePayload,
    ConversationScope,
)


def _agent_scope() -> ConversationScope:
    return ConversationScope(
        conversation_id=101,
        user_id=7,
        mode=ConversationMode.AGENT,
        knowledge_base_id=11,
    )


def _thread() -> AgentThreadIdentity:
    return AgentThreadIdentity(
        thread_id="thread-101",
        conversation_id=101,
    )


def test_conversation_scope_freezes_user_mode_and_knowledge_base() -> None:
    scope = _agent_scope()

    assert scope.user_id == 7
    assert scope.mode == ConversationMode.AGENT
    assert scope.knowledge_base_id == 11

    with pytest.raises(ValidationError):
        scope.knowledge_base_id = 12  # type: ignore[misc]

    with pytest.raises(ValidationError):
        ConversationScope.model_validate(
            {
                "conversation_id": 101,
                "user_id": 7,
                "mode": "agent",
                "knowledge_base_id": 11,
                "unexpected": True,
            }
        )


def test_conversation_message_only_allows_user_visible_roles() -> None:
    user_message = ConversationMessagePayload(
        role=ConversationMessageRole.USER,
        content="  比较方案 A 和方案 B  ",
    )

    assert user_message.content == "比较方案 A 和方案 B"

    with pytest.raises(ValidationError):
        ConversationMessagePayload.model_validate(
            {
                "role": "tool",
                "content": "internal tool result",
            }
        )


def test_agent_state_is_framework_neutral_and_json_round_trippable() -> None:
    state = AgentState(
        conversation=_agent_scope(),
        thread=_thread(),
        agent_run_id=9001,
        status=AgentStateStatus.RUNNING,
        messages=(
            ConversationMessagePayload(
                role=ConversationMessageRole.USER,
                content="比较方案 A 和方案 B",
            ),
        ),
        task="  比较两个技术方案并给出推荐  ",
    )

    payload = state.model_dump_json()
    restored = AgentState.model_validate_json(payload)

    assert restored == state
    assert restored.state_schema_version == AGENT_STATE_SCHEMA_VERSION
    assert restored.task == "比较两个技术方案并给出推荐"
    assert "langgraph" not in payload.lower()
    assert "langchain" not in payload.lower()


def test_agent_state_rejects_rag_conversation() -> None:
    rag_scope = ConversationScope(
        conversation_id=101,
        user_id=7,
        mode=ConversationMode.RAG,
        knowledge_base_id=11,
    )

    with pytest.raises(
        ValidationError,
        match="AgentState requires an agent-mode conversation",
    ):
        AgentState(
            conversation=rag_scope,
            thread=_thread(),
        )


def test_agent_state_rejects_thread_from_another_conversation() -> None:
    with pytest.raises(
        ValidationError,
        match="thread conversation_id must match conversation scope",
    ):
        AgentState(
            conversation=_agent_scope(),
            thread=AgentThreadIdentity(
                thread_id="thread-other",
                conversation_id=202,
            ),
        )


def test_agent_state_rejects_unknown_schema_version() -> None:
    with pytest.raises(ValidationError):
        AgentState.model_validate(
            {
                "state_schema_version": "2.0",
                "conversation": _agent_scope().model_dump(mode="json"),
                "thread": _thread().model_dump(mode="json"),
            }
        )
