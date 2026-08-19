from collections.abc import Sequence
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from app.agent.context import ToolExecutionContext
from app.agent.version_snapshot import (
    AgentEvaluationVersionContext,
    AgentRuntimeVersionSnapshot,
)
from app.agent.model_response import LLMToolCall, LLMToolExchange, LLMToolResponse
from app.agent.run_event import AgentToolCallEvent
from app.agent.native_agent import AgentRepeatedToolCallError, NativeAgentRunner
from app.agent.tools.base import BaseAgentTool, ToolContract, ToolRiskLevel
from app.constants.agent_run_status import AgentRunStatus
from app.constants.agent_tool_call_status import AgentToolCallStatus
from app.constants.user_role import UserRole
from app.models.database.agent_run import AgentRun
from app.models.database.agent_tool_call import AgentToolCall
from app.models.database.knowledge_base import KnowledgeBase
from app.models.database.user import User
from app.repositories.agent_run_repository import AgentRunRepository
from app.repositories.agent_tool_call_repository import AgentToolCallRepository
from app.services.agent_execution_service import AgentExecutionService


class EchoInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1)


class EchoOutput(BaseModel):
    query: str
    agent_run_id: int


class EchoTool(BaseAgentTool[EchoInput, EchoOutput]):
    name = "echo_tool"
    version = "1.2.3"
    description = "Echo validated input for persistence tests."
    risk_level = ToolRiskLevel.READ_ONLY
    input_model = EchoInput
    output_model = EchoOutput

    def __init__(self) -> None:
        self.contexts: list[ToolExecutionContext] = []

    def execute(
        self,
        db: Session,
        context: ToolExecutionContext,
        tool_input: EchoInput,
    ) -> EchoOutput:
        self.contexts.append(context)
        assert isinstance(context.agent_run_id, int)
        return EchoOutput(
            query=tool_input.query,
            agent_run_id=context.agent_run_id,
        )


class UnexpectedErrorTool(EchoTool):
    name = "unexpected_error"

    def execute(
        self,
        db: Session,
        context: ToolExecutionContext,
        tool_input: EchoInput,
    ) -> EchoOutput:
        raise RuntimeError("private-provider-detail")


class FakeToolCallingLLM:
    def __init__(
        self,
        responses: Sequence[LLMToolResponse | Exception],
    ) -> None:
        self.responses = list(responses)
        self.contexts: list[dict[str, Any]] = []

    def chat_with_tool_history(
        self,
        *,
        message: str,
        tool_contracts: Sequence[ToolContract],
        history: Sequence[LLMToolExchange],
    ) -> LLMToolResponse:
        self.contexts.append(
            {
                "message": message,
                "history": list(history),
            }
        )
        if not self.responses:
            raise AssertionError("unexpected extra model call")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _tool_call(
    call_id: str = "call_001",
    name: str = "echo_tool",
    query: str = "部署说明",
) -> LLMToolCall:
    return LLMToolCall(
        id=call_id,
        name=name,
        arguments_json=f'{{"query":"{query}"}}',
    )


def _create_scope(db: Session) -> tuple[User, KnowledgeBase]:
    user = User(
        email="agent-run-owner@example.com",
        password_hash="test-password-hash",
        role=UserRole.USER.value,
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    knowledge_base = KnowledgeBase(
        owner_id=user.id,
        name="Agent Run KB",
        description="Agent persistence tests",
    )
    db.add(knowledge_base)
    db.commit()
    db.refresh(knowledge_base)
    return user, knowledge_base


def _context(user: User, kb: KnowledgeBase, request_id: str) -> ToolExecutionContext:
    return ToolExecutionContext(
        user_id=user.id,
        role=UserRole.USER,
        knowledge_base_id=kb.id,
        request_id=request_id,
    )


def _service(
    *,
    llm: FakeToolCallingLLM,
    tool: BaseAgentTool[Any, Any],
    max_turns: int = 4,
) -> AgentExecutionService:
    runner = NativeAgentRunner(
        llm_service=llm,
        tools=[tool],
        max_turns=max_turns,
    )
    return AgentExecutionService(
        agent_runner=runner,
        agent_run_repository=AgentRunRepository(),
        tool_call_repository=AgentToolCallRepository(),
        model_provider="test-provider",
        model_name="test-model",
        version_snapshot=AgentRuntimeVersionSnapshot(
            agent_version="2.0.0-test",
            prompt_version="1.0.0-test",
            toolset_version="toolset-v1:test",
            retrieval_config_version="retrieval-v1:test",
        ),
    )


def _latest_run(db: Session, request_id: str) -> AgentRun:
    run = AgentRunRepository().find_latest_by_request_id(db, request_id)
    assert run is not None
    return run


def test_agent_execution_persists_direct_success_without_sensitive_content(
    db: Session,
) -> None:
    user, kb = _create_scope(db)
    request_id = "agent-run-direct"
    service = _service(
        llm=FakeToolCallingLLM(
            [LLMToolResponse(content="这是最终回答。")]
        ),
        tool=EchoTool(),
    )

    result = service.run(
        db=db,
        context=_context(user, kb, request_id),
        message="包含企业敏感问题正文",
    )

    run = _latest_run(db, request_id)
    assert result.answer == "这是最终回答。"
    assert run.status == AgentRunStatus.SUCCEEDED.value
    assert run.user_id == user.id
    assert run.knowledge_base_id == kb.id
    assert run.model_provider == "test-provider"
    assert run.model_name == "test-model"
    assert run.agent_version == "2.0.0-test"
    assert run.prompt_version == "1.0.0-test"
    assert run.toolset_version == "toolset-v1:test"
    assert run.retrieval_config_version == "retrieval-v1:test"
    assert run.eval_dataset_version is None
    assert run.evaluator_version is None
    assert run.tool_call_count == 0
    assert run.finished_at is not None
    assert run.error_type is None

    # C1 数据模型不保存 Prompt、回答、Tool 参数或 Tool Result 正文。
    assert not hasattr(run, "prompt")
    assert not hasattr(run, "message")
    assert not hasattr(run, "answer")


def test_agent_execution_persists_tool_call_and_injects_agent_run_id(
    db: Session,
) -> None:
    user, kb = _create_scope(db)
    request_id = "agent-run-tool-success"
    tool = EchoTool()
    service = _service(
        llm=FakeToolCallingLLM(
            [
                LLMToolResponse(tool_calls=[_tool_call()]),
                LLMToolResponse(content="根据 Tool 结果完成回答。"),
            ]
        ),
        tool=tool,
    )

    result = service.run(
        db=db,
        context=_context(user, kb, request_id),
        message="查询部署说明",
    )

    run = _latest_run(db, request_id)
    tool_calls = AgentToolCallRepository().find_all_by_agent_run_id(
        db,
        run.id,
    )

    assert result.tool_call_count == 1
    assert run.status == AgentRunStatus.SUCCEEDED.value
    assert run.tool_call_count == 1
    assert len(tool_calls) == 1

    persisted = tool_calls[0]
    assert persisted.provider_call_id == "call_001"
    assert persisted.tool_name == "echo_tool"
    assert persisted.tool_version == "1.2.3"
    assert persisted.status == AgentToolCallStatus.SUCCEEDED.value
    assert persisted.duration_ms is not None
    assert persisted.duration_ms >= 0
    assert persisted.finished_at is not None
    assert persisted.error_type is None

    assert len(tool.contexts) == 1
    assert tool.contexts[0].agent_run_id == run.id
    assert not hasattr(persisted, "arguments_json")
    assert not hasattr(persisted, "content_json")


def test_tool_failure_can_be_persisted_while_agent_run_recovers(
    db: Session,
) -> None:
    user, kb = _create_scope(db)
    request_id = "agent-run-tool-failure"
    service = _service(
        llm=FakeToolCallingLLM(
            [
                LLMToolResponse(
                    tool_calls=[
                        _tool_call(name="unexpected_error")
                    ]
                ),
                LLMToolResponse(content="Tool 失败后的安全降级回答。"),
            ]
        ),
        tool=UnexpectedErrorTool(),
    )

    service.run(
        db=db,
        context=_context(user, kb, request_id),
        message="查询资料",
    )

    run = _latest_run(db, request_id)
    tool_calls = AgentToolCallRepository().find_all_by_agent_run_id(
        db,
        run.id,
    )

    assert run.status == AgentRunStatus.SUCCEEDED.value
    assert run.tool_call_count == 1
    assert tool_calls[0].status == AgentToolCallStatus.FAILED.value
    assert tool_calls[0].error_type == "execution_failed"
    assert "private-provider-detail" not in (tool_calls[0].error_type or "")


def test_agent_loop_error_marks_run_failed_with_safe_error_type(
    db: Session,
) -> None:
    user, kb = _create_scope(db)
    request_id = "agent-run-loop-failed"
    service = _service(
        llm=FakeToolCallingLLM(
            [
                LLMToolResponse(
                    tool_calls=[_tool_call(call_id="call_001")]
                ),
                LLMToolResponse(
                    tool_calls=[_tool_call(call_id="call_002")]
                ),
            ]
        ),
        tool=EchoTool(),
    )

    with pytest.raises(AgentRepeatedToolCallError):
        service.run(
            db=db,
            context=_context(user, kb, request_id),
            message="重复 Tool 测试",
        )

    run = _latest_run(db, request_id)
    assert run.status == AgentRunStatus.FAILED.value
    assert run.error_type == "repeated_tool_call"
    assert run.finished_at is not None
    assert run.tool_call_count == 1


def test_unexpected_model_error_marks_run_failed_without_error_detail(
    db: Session,
) -> None:
    user, kb = _create_scope(db)
    request_id = "agent-run-model-failed"
    service = _service(
        llm=FakeToolCallingLLM(
            [RuntimeError("private-model-provider-detail")]
        ),
        tool=EchoTool(),
    )

    with pytest.raises(RuntimeError, match="private-model-provider-detail"):
        service.run(
            db=db,
            context=_context(user, kb, request_id),
            message="模型异常",
        )

    run = _latest_run(db, request_id)
    assert run.status == AgentRunStatus.FAILED.value
    assert run.error_type == "RuntimeError"
    assert "private-model-provider-detail" not in (run.error_type or "")


def test_stream_close_marks_open_run_and_tool_call_failed_for_c1(
    db: Session,
) -> None:
    """C1 尚无 cancelled 状态，流提前关闭暂记 failed + stream_cancelled。"""

    user, kb = _create_scope(db)
    request_id = "agent-run-stream-cancelled"
    tool = EchoTool()
    service = _service(
        llm=FakeToolCallingLLM(
            [LLMToolResponse(tool_calls=[_tool_call()])]
        ),
        tool=tool,
    )

    stream = service.run_events(
        db=db,
        context=_context(user, kb, request_id),
        message="流式取消测试",
    )

    first = next(stream)
    assert first.type == "status"
    second = next(stream)
    assert isinstance(second, AgentToolCallEvent)

    run = _latest_run(db, request_id)
    persisted_calls = AgentToolCallRepository().find_all_by_agent_run_id(
        db,
        run.id,
    )
    assert run.status == AgentRunStatus.RUNNING.value
    assert persisted_calls[0].status == AgentToolCallStatus.RUNNING.value

    stream.close()
    db.expire_all()

    run = _latest_run(db, request_id)
    persisted_calls = AgentToolCallRepository().find_all_by_agent_run_id(
        db,
        run.id,
    )
    assert run.status == AgentRunStatus.FAILED.value
    assert run.error_type == "stream_cancelled"
    assert persisted_calls[0].status == AgentToolCallStatus.FAILED.value
    assert persisted_calls[0].error_type == "stream_cancelled"
    assert tool.contexts == []


def test_agent_execution_persists_eval_version_context(
    db: Session,
) -> None:
    """Eval Run 额外固化 Dataset / Evaluator 版本，普通 Run 不受影响。"""

    user, kb = _create_scope(db)
    request_id = "agent-run-eval-version"
    service = _service(
        llm=FakeToolCallingLLM([LLMToolResponse(content="评估回答")]),
        tool=EchoTool(),
    )

    service.run(
        db=db,
        context=_context(user, kb, request_id),
        message="评估问题",
        evaluation_version=AgentEvaluationVersionContext(
            dataset_version="1.5.0",
            evaluator_version="1.4.0",
        ),
    )

    run = _latest_run(db, request_id)
    assert run.eval_dataset_version == "1.5.0"
    assert run.evaluator_version == "1.4.0"
