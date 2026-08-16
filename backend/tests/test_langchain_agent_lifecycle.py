from collections.abc import Sequence
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from app.agent.context import ToolExecutionContext
from app.agent.frameworks.langchain.runner import (
    LangChainAgentRepeatedToolCallError,
    LangChainAgentResult,
)
from app.agent.model_response import LLMToolCall
from app.agent.run_observer import AgentRunObserver
from app.agent.tools.base import BaseAgentTool, ToolRiskLevel
from app.agent.version_snapshot import (
    AgentEvaluationVersionContext,
    AgentRuntimeVersionSnapshot,
)
from app.constants.agent_run_status import AgentRunStatus
from app.constants.agent_tool_call_status import AgentToolCallStatus
from app.constants.user_role import UserRole
from app.models.database.agent_run import AgentRun
from app.models.database.knowledge_base import KnowledgeBase
from app.models.database.user import User
from app.repositories.agent_run_repository import AgentRunRepository
from app.repositories.agent_tool_call_repository import AgentToolCallRepository
from app.services.langchain_agent_execution_service import (
    LangChainAgentExecutionService,
)


class EchoInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1)


class EchoOutput(BaseModel):
    query: str


class EchoTool(BaseAgentTool[EchoInput, EchoOutput]):
    name = "echo_tool"
    version = "2.1.0"
    description = "Echo Tool for LangChain lifecycle tests."
    risk_level = ToolRiskLevel.READ_ONLY
    input_model = EchoInput
    output_model = EchoOutput

    def execute(
        self,
        db: Session,
        context: ToolExecutionContext,
        tool_input: EchoInput,
    ) -> EchoOutput:
        return EchoOutput(query=tool_input.query)


class ScriptedLangChainLifecycleRunner:
    RUNNER_VERSION = "1.3.0"

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.tools = [EchoTool()]
        self.tool_contracts = [tool.get_contract() for tool in self.tools]
        self.contexts: list[ToolExecutionContext] = []

    def run(
        self,
        *,
        db: Session,
        context: ToolExecutionContext,
        message: str,
        observer: AgentRunObserver | None = None,
        execution_observer=None,
    ) -> LangChainAgentResult:
        self.contexts.append(context)
        assert context.agent_run_id is not None

        if self.mode == "direct":
            answer = "直接回答。"
            if observer is not None:
                observer.on_final_answer(answer)
            return LangChainAgentResult(
                answer=answer,
                turns=1,
                tool_call_count=0,
            )

        call = LLMToolCall(
            id="call-lc-001",
            name="echo_tool",
            arguments_json='{"query":"部署说明"}',
        )
        if observer is not None:
            observer.on_tool_call_requested(call)

        if self.mode == "guard_reject":
            raise LangChainAgentRepeatedToolCallError(
                "repeated tool call: echo_tool"
            )

        assert execution_observer is not None
        execution_observer.on_tool_execution_started(
            call_id=call.id,
            tool_name=call.name,
        )

        ok = self.mode != "tool_failure"
        error_code = None if ok else "execution_failed"
        execution_observer.on_tool_execution_finished(
            call_id=call.id,
            tool_name=call.name,
            ok=ok,
            error_code=error_code,
            duration_ms=7,
        )
        if observer is not None:
            observer.on_tool_result(
                call_id=call.id,
                tool_name=call.name,
                ok=ok,
                error_code=error_code,
                evidence_refs=[],
            )
            observer.on_final_answer("Tool 后回答。")

        return LangChainAgentResult(
            answer="Tool 后回答。",
            turns=2,
            tool_call_count=1,
        )


def _create_scope(db: Session) -> tuple[User, KnowledgeBase]:
    user = User(
        email="langchain-lifecycle@example.com",
        password_hash="test-password-hash",
        role=UserRole.USER.value,
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    knowledge_base = KnowledgeBase(
        owner_id=user.id,
        name="LangChain Lifecycle KB",
        description="candidate lifecycle tests",
    )
    db.add(knowledge_base)
    db.commit()
    db.refresh(knowledge_base)
    return user, knowledge_base


def _context(
    user: User,
    knowledge_base: KnowledgeBase,
    request_id: str,
) -> ToolExecutionContext:
    return ToolExecutionContext(
        user_id=user.id,
        role=UserRole.USER,
        knowledge_base_id=knowledge_base.id,
        request_id=request_id,
    )


def _service(
    runner: ScriptedLangChainLifecycleRunner,
) -> LangChainAgentExecutionService:
    return LangChainAgentExecutionService(
        agent_runner=runner,  # type: ignore[arg-type]
        agent_run_repository=AgentRunRepository(),
        tool_call_repository=AgentToolCallRepository(),
        model_provider="test-provider",
        model_name="test-model",
        version_snapshot=AgentRuntimeVersionSnapshot(
            agent_version="langchain-v1:1.3.0",
            prompt_version="1.0.0-test",
            toolset_version="toolset-v1:test",
            retrieval_config_version="retrieval-v1:test",
        ),
    )


def _latest_run(db: Session, request_id: str) -> AgentRun:
    run = AgentRunRepository().find_latest_by_request_id(db, request_id)
    assert run is not None
    return run


def test_langchain_lifecycle_persists_direct_success_and_eval_version(
    db: Session,
) -> None:
    user, knowledge_base = _create_scope(db)
    request_id = "langchain-run-direct"
    runner = ScriptedLangChainLifecycleRunner("direct")

    result = _service(runner).run(
        db=db,
        context=_context(user, knowledge_base, request_id),
        message="你能做什么？",
        evaluation_version=AgentEvaluationVersionContext(
            dataset_version="1.5.0",
            evaluator_version="1.4.0",
        ),
    )

    run = _latest_run(db, request_id)
    assert result.answer == "直接回答。"
    assert run.status == AgentRunStatus.SUCCEEDED.value
    assert run.agent_version == "langchain-v1:1.3.0"
    assert run.prompt_version == "1.0.0-test"
    assert run.toolset_version == "toolset-v1:test"
    assert run.retrieval_config_version == "retrieval-v1:test"
    assert run.eval_dataset_version == "1.5.0"
    assert run.evaluator_version == "1.4.0"
    assert run.tool_call_count == 0
    assert run.error_type is None
    assert runner.contexts[0].agent_run_id == run.id

    # 生命周期表继续不保存 Prompt / Answer / Tool 参数正文。
    assert not hasattr(run, "prompt")
    assert not hasattr(run, "answer")


def test_langchain_lifecycle_persists_only_actual_tool_execution(
    db: Session,
) -> None:
    user, knowledge_base = _create_scope(db)
    request_id = "langchain-run-tool"
    runner = ScriptedLangChainLifecycleRunner("tool_success")

    _service(runner).run(
        db=db,
        context=_context(user, knowledge_base, request_id),
        message="查询部署说明",
    )

    run = _latest_run(db, request_id)
    tool_calls = AgentToolCallRepository().find_all_by_agent_run_id(
        db,
        run.id,
    )

    assert run.status == AgentRunStatus.SUCCEEDED.value
    assert run.tool_call_count == 1
    assert len(tool_calls) == 1
    persisted = tool_calls[0]
    assert persisted.provider_call_id == "call-lc-001"
    assert persisted.tool_name == "echo_tool"
    assert persisted.tool_version == "2.1.0"
    assert persisted.status == AgentToolCallStatus.SUCCEEDED.value
    assert persisted.duration_ms == 7
    assert persisted.error_type is None
    assert persisted.finished_at is not None
    assert not hasattr(persisted, "arguments_json")


def test_langchain_lifecycle_can_record_tool_failure_while_run_recovers(
    db: Session,
) -> None:
    user, knowledge_base = _create_scope(db)
    request_id = "langchain-run-tool-failure"
    runner = ScriptedLangChainLifecycleRunner("tool_failure")

    _service(runner).run(
        db=db,
        context=_context(user, knowledge_base, request_id),
        message="执行失败 Tool",
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


def test_guard_rejected_model_request_is_observable_but_not_persisted_as_execution(
    db: Session,
) -> None:
    user, knowledge_base = _create_scope(db)
    request_id = "langchain-run-guard-reject"
    runner = ScriptedLangChainLifecycleRunner("guard_reject")

    with pytest.raises(LangChainAgentRepeatedToolCallError):
        _service(runner).run(
            db=db,
            context=_context(user, knowledge_base, request_id),
            message="重复调用",
        )

    run = _latest_run(db, request_id)
    tool_calls = AgentToolCallRepository().find_all_by_agent_run_id(
        db,
        run.id,
    )
    assert run.status == AgentRunStatus.FAILED.value
    assert run.error_type == "repeated_tool_call"
    assert run.tool_call_count == 0
    assert tool_calls == []
