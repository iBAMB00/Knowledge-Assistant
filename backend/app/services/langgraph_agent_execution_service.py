"""Production lifecycle bridge for the v2.3 LangGraph Stateful Runtime."""

from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.agent.context import ToolExecutionContext
from app.agent.frameworks.langgraph.runner import LangGraphStatefulRunner
from app.agent.hitl import AgentApprovalStateError, AgentInterruptRequired
from app.agent.native_agent import AgentLoopError, NativeAgentResult
from app.agent.run_control import AgentRunCancellationError
from app.agent.run_event import (
    AgentMessageEvent,
    AgentRunEvent,
    AgentToolCallEvent,
    AgentToolResultEvent,
)
from app.agent.run_observer import AgentRunObserver
from app.agent.state import AgentState, AgentThreadIdentity
from app.agent.version_snapshot import (
    AgentEvaluationVersionContext,
    AgentRuntimeVersionSnapshot,
)
from app.constants.agent_run_status import AgentRunStatus
from app.constants.agent_state_status import AgentStateStatus
from app.constants.conversation_mode import ConversationMode
from app.repositories.agent_run_repository import AgentRunRepository
from app.repositories.agent_tool_call_repository import AgentToolCallRepository
from app.schemas.conversation_contract import ConversationScope
from app.services.agent_checkpoint_service import AgentCheckpointService
from app.services.agent_execution_service import AgentExecutionService
from app.services.agent_hitl_service import AgentHITLService
from app.services.agent_recovery_service import AgentRecoveryService
from app.services.conversation_history_context_provider import ConversationHistoryContextProvider


class AgentStatefulRunConflictError(RuntimeError):
    """当前 Conversation 的 Thread 仍有未结束/终止状态，不能启动新任务。"""


class LangGraphAgentExecutionService(AgentExecutionService):
    """
    把 LangGraph Stateful Runner 接入既有 AgentRun / ToolCall 生命周期事实。

    Runner 继续负责 Graph/Checkpoint/Resume/HITL/Cancel；本 Service 只负责：
    - fresh run 绑定已校验 Conversation；
    - 每次 fresh/resume 都创建独立 AgentRun attempt；
    - 安全事件映射到 AgentToolCall；
    - WAITING -> AgentRun.interrupted，Cancel -> AgentRun.cancelled。
    """

    APPROVAL_REQUIRED_ERROR = "approval_required"
    CANCELLED_ERROR = "agent_cancelled"

    def __init__(
        self,
        *,
        agent_runner: LangGraphStatefulRunner,
        agent_run_repository: AgentRunRepository,
        tool_call_repository: AgentToolCallRepository,
        model_provider: str,
        model_name: str,
        version_snapshot: AgentRuntimeVersionSnapshot,
        checkpoint_service: AgentCheckpointService,
        recovery_service: AgentRecoveryService,
        hitl_service: AgentHITLService,
        conversation_history_provider: ConversationHistoryContextProvider | None = None,
    ) -> None:
        super().__init__(
            agent_runner=agent_runner,
            agent_run_repository=agent_run_repository,
            tool_call_repository=tool_call_repository,
            model_provider=model_provider,
            model_name=model_name,
            version_snapshot=version_snapshot,
            conversation_history_provider=conversation_history_provider,
        )
        self.agent_runner = agent_runner
        self.checkpoint_service = checkpoint_service
        self.recovery_service = recovery_service
        self.hitl_service = hitl_service

    def run(
        self,
        *,
        db: Session,
        context: ToolExecutionContext,
        message: str,
        observer: AgentRunObserver | None = None,
        evaluation_version: AgentEvaluationVersionContext | None = None,
    ) -> NativeAgentResult:
        final_result: NativeAgentResult | None = None
        for event in self.run_events(
            db=db,
            context=context,
            message=message,
            observer=observer,
            evaluation_version=evaluation_version,
        ):
            if isinstance(event, AgentMessageEvent):
                final_result = NativeAgentResult(
                    answer=event.content,
                    turns=event.turns,
                    tool_call_count=event.tool_call_count,
                )
        if final_result is None:
            raise RuntimeError("langgraph run completed without final answer")
        return final_result

    def run_events(
        self,
        *,
        db: Session,
        context: ToolExecutionContext,
        message: str,
        observer: AgentRunObserver | None = None,
        evaluation_version: AgentEvaluationVersionContext | None = None,
    ) -> Iterator[AgentRunEvent]:
        normalized_message = self._normalize_message(message)
        state = self._build_fresh_state(db=db, context=context)
        supporting_context = self._load_conversation_history(
            db=db,
            context=context,
            current_message=normalized_message,
        )

        def stream_factory(run_context: ToolExecutionContext):
            kwargs = dict(
                db=db,
                context=run_context,
                message=normalized_message,
                state=state,
                observer=observer,
            )
            if supporting_context:
                kwargs["supporting_context"] = supporting_context
            return self.agent_runner.run_events(**kwargs)

        yield from self._execute_attempt(
            db=db,
            context=context,
            evaluation_version=evaluation_version,
            stream_factory=stream_factory,
        )

    def resume(
        self,
        *,
        db: Session,
        context: ToolExecutionContext,
        thread_id: str,
        observer: AgentRunObserver | None = None,
        evaluation_version: AgentEvaluationVersionContext | None = None,
    ) -> NativeAgentResult:
        final_result: NativeAgentResult | None = None
        for event in self.resume_events(
            db=db,
            context=context,
            thread_id=thread_id,
            observer=observer,
            evaluation_version=evaluation_version,
        ):
            if isinstance(event, AgentMessageEvent):
                final_result = NativeAgentResult(
                    answer=event.content,
                    turns=event.turns,
                    tool_call_count=event.tool_call_count,
                )
        if final_result is None:
            raise RuntimeError("langgraph resume completed without final answer")
        return final_result

    def resume_events(
        self,
        *,
        db: Session,
        context: ToolExecutionContext,
        thread_id: str,
        observer: AgentRunObserver | None = None,
        evaluation_version: AgentEvaluationVersionContext | None = None,
    ) -> Iterator[AgentRunEvent]:
        normalized_thread_id = thread_id.strip()
        if not normalized_thread_id:
            raise ValueError("thread_id cannot be empty")

        payload = self.checkpoint_service.load_latest_for_scope(
            db,
            thread_id=normalized_thread_id,
            user_id=context.user_id,
            knowledge_base_id=context.knowledge_base_id,
        )
        if payload is None:
            # 使用既有 Recovery error contract，避免暴露 Thread 存在性。
            self.recovery_service.load_resume_checkpoint(
                db,
                thread_id=normalized_thread_id,
                user_id=context.user_id,
                knowledge_base_id=context.knowledge_base_id,
            )
            raise AssertionError("unreachable")

        scoped_context = context.model_copy(
            update={
                "conversation_id": (
                    payload.agent_state.conversation.conversation_id
                )
            }
        )

        resume_task = payload.agent_state.task or ""
        supporting_context = (
            self._load_conversation_history(
                db=db,
                context=scoped_context,
                current_message=resume_task,
            )
            if resume_task
            else ()
        )

        if payload.agent_state.status is AgentStateStatus.RUNNING:
            # 在创建新 AgentRun 前先做一次完整恢复资格校验。
            self.recovery_service.load_resume_checkpoint(
                db,
                thread_id=normalized_thread_id,
                user_id=context.user_id,
                knowledge_base_id=context.knowledge_base_id,
            )
            def factory(run_context: ToolExecutionContext):
                kwargs = dict(
                    db=db,
                    context=run_context,
                    thread_id=normalized_thread_id,
                    observer=observer,
                )
                if supporting_context:
                    kwargs["supporting_context"] = supporting_context
                return self.agent_runner.resume_events(**kwargs)
        elif payload.agent_state.status is AgentStateStatus.WAITING:
            # 只有所有 pending approvals 已 durable 批准后才允许真正续跑。
            self.hitl_service.load_approved_checkpoint(
                db,
                thread_id=normalized_thread_id,
                user_id=context.user_id,
                knowledge_base_id=context.knowledge_base_id,
            )
            def factory(run_context: ToolExecutionContext):
                kwargs = dict(
                    db=db,
                    context=run_context,
                    thread_id=normalized_thread_id,
                    observer=observer,
                )
                if supporting_context:
                    kwargs["supporting_context"] = supporting_context
                return self.agent_runner.resume_after_approval_events(**kwargs)
        else:
            from app.agent.checkpoint import AgentResumeStateError

            raise AgentResumeStateError(
                "agent thread is not resumable from status: "
                f"{payload.agent_state.status.value}"
            )

        yield from self._execute_attempt(
            db=db,
            context=scoped_context,
            evaluation_version=evaluation_version,
            stream_factory=factory,
        )

    def _build_fresh_state(
        self,
        *,
        db: Session,
        context: ToolExecutionContext,
    ) -> AgentState:
        conversation_id = context.conversation_id
        if conversation_id is None:
            raise ValueError("langgraph runtime requires conversation_id")

        thread_id = f"conversation:{conversation_id}"
        latest = self.checkpoint_service.load_latest_for_scope(
            db,
            thread_id=thread_id,
            user_id=context.user_id,
            knowledge_base_id=context.knowledge_base_id,
        )
        if latest is not None and latest.agent_state.status in {
            AgentStateStatus.RUNNING,
            AgentStateStatus.WAITING,
        }:
            raise AgentStatefulRunConflictError(
                "agent thread still has an active execution; resume or cancel it first"
            )

        return AgentState(
            conversation=ConversationScope(
                conversation_id=conversation_id,
                user_id=context.user_id,
                mode=ConversationMode.AGENT,
                knowledge_base_id=context.knowledge_base_id,
            ),
            thread=AgentThreadIdentity(
                thread_id=thread_id,
                conversation_id=conversation_id,
            ),
        )

    def _execute_attempt(
        self,
        *,
        db: Session,
        context: ToolExecutionContext,
        evaluation_version: AgentEvaluationVersionContext | None,
        stream_factory: Callable[
            [ToolExecutionContext], Iterator[AgentRunEvent]
        ],
    ) -> Iterator[AgentRunEvent]:
        agent_run = self._start_run(
            db=db,
            context=context,
            evaluation_version=evaluation_version,
        )
        run_context = context.model_copy(
            update={"agent_run_id": agent_run.id}
        )
        event_stream: Iterator[AgentRunEvent] | None = None
        open_tool_calls: dict[str, int] = {}
        tool_call_count = 0
        completed = False

        try:
            event_stream = stream_factory(run_context)
            for event in event_stream:
                if isinstance(event, AgentToolCallEvent):
                    tool_call = self._start_tool_call(
                        db=db,
                        agent_run_id=agent_run.id,
                        event=event,
                    )
                    open_tool_calls[event.call_id] = tool_call.id
                    tool_call_count += 1
                elif isinstance(event, AgentToolResultEvent):
                    tool_call_id = open_tool_calls.pop(event.call_id, None)
                    if tool_call_id is None:
                        raise RuntimeError(
                            "tool result has no persisted tool call"
                        )
                    self._finish_tool_call(
                        db=db,
                        tool_call_id=tool_call_id,
                        event=event,
                    )
                elif isinstance(event, AgentMessageEvent):
                    self._succeed_run(
                        db=db,
                        agent_run_id=agent_run.id,
                        tool_call_count=tool_call_count,
                    )
                    completed = True
                yield event

            if not completed:
                raise RuntimeError(
                    "langgraph event stream completed without final answer"
                )

        except GeneratorExit:
            if not completed:
                self._safe_fail_open_tool_calls(
                    db=db,
                    open_tool_calls=open_tool_calls,
                    error_type="stream_cancelled",
                )
                self._safe_fail_run(
                    db=db,
                    agent_run_id=agent_run.id,
                    tool_call_count=tool_call_count,
                    error_type="stream_cancelled",
                )
            raise

        except AgentInterruptRequired:
            self._safe_fail_open_tool_calls(
                db=db,
                open_tool_calls=open_tool_calls,
                error_type=self.APPROVAL_REQUIRED_ERROR,
            )
            self._safe_finish_run_status(
                db=db,
                agent_run_id=agent_run.id,
                status=AgentRunStatus.INTERRUPTED,
                tool_call_count=tool_call_count,
                error_type=self.APPROVAL_REQUIRED_ERROR,
            )
            raise

        except AgentRunCancellationError:
            self._safe_fail_open_tool_calls(
                db=db,
                open_tool_calls=open_tool_calls,
                error_type=self.CANCELLED_ERROR,
            )
            self._safe_finish_run_status(
                db=db,
                agent_run_id=agent_run.id,
                status=AgentRunStatus.CANCELLED,
                tool_call_count=tool_call_count,
                error_type=self.CANCELLED_ERROR,
            )
            raise

        except AgentLoopError as exc:
            self._safe_fail_open_tool_calls(
                db=db,
                open_tool_calls=open_tool_calls,
                error_type=exc.code,
            )
            self._safe_fail_run(
                db=db,
                agent_run_id=agent_run.id,
                tool_call_count=tool_call_count,
                error_type=exc.code,
            )
            raise

        except Exception as exc:
            error_type = type(exc).__name__
            self._safe_fail_open_tool_calls(
                db=db,
                open_tool_calls=open_tool_calls,
                error_type=error_type,
            )
            self._safe_fail_run(
                db=db,
                agent_run_id=agent_run.id,
                tool_call_count=tool_call_count,
                error_type=error_type,
            )
            raise

        finally:
            self._close_iterator(event_stream)

    def _finish_run_status(
        self,
        *,
        db: Session,
        agent_run_id: int,
        status: AgentRunStatus,
        tool_call_count: int,
        error_type: str | None,
    ) -> None:
        db.rollback()
        agent_run = self._get_run(db=db, agent_run_id=agent_run_id)
        agent_run.status = status.value
        agent_run.tool_call_count = tool_call_count
        agent_run.error_type = (
            self._normalize_error_type(error_type)
            if error_type is not None
            else None
        )
        agent_run.finished_at = datetime.now(timezone.utc)
        db.commit()

    def _safe_finish_run_status(self, **kwargs: Any) -> None:
        try:
            self._finish_run_status(**kwargs)
        except Exception:
            # 与 Native bridge 一致：观测事实写入失败不能吞掉原始 Runtime 控制异常。
            db = kwargs.get("db")
            if isinstance(db, Session):
                db.rollback()
