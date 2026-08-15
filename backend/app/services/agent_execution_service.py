import logging
from collections.abc import Iterator
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.agent.context import ToolExecutionContext
from app.agent.native_agent import (
    AgentLoopError,
    NativeAgentResult,
    NativeAgentRunner,
)
from app.agent.run_event import (
    AgentMessageEvent,
    AgentRunEvent,
    AgentToolCallEvent,
    AgentToolResultEvent,
)
from app.agent.run_observer import AgentRunObserver
from app.constants.agent_run_status import AgentRunStatus
from app.constants.agent_tool_call_status import AgentToolCallStatus
from app.models.database.agent_run import AgentRun
from app.models.database.agent_tool_call import AgentToolCall
from app.repositories.agent_run_repository import AgentRunRepository
from app.repositories.agent_tool_call_repository import AgentToolCallRepository


logger = logging.getLogger(__name__)


class AgentExecutionService:
    """
    将 Native Agent Runtime 映射为可持久化的运行生命周期。

    NativeAgentRunner 继续只负责 Agent Loop；本 Service 负责：
    - 创建/结束 AgentRun
    - 创建/结束 AgentToolCall
    - 事务提交与失败回滚
    - 将 agent_run_id 注入新的可信 ToolExecutionContext

    C1 不保存 Prompt、隐藏推理、Tool 参数或 Tool Result 正文。
    """

    def __init__(
        self,
        *,
        agent_runner: NativeAgentRunner,
        agent_run_repository: AgentRunRepository,
        tool_call_repository: AgentToolCallRepository,
        model_provider: str,
        model_name: str,
    ) -> None:
        normalized_provider = model_provider.strip()
        normalized_model_name = model_name.strip()

        if not normalized_provider:
            raise ValueError("model_provider cannot be empty")
        if not normalized_model_name:
            raise ValueError("model_name cannot be empty")

        self.agent_runner = agent_runner
        self.agent_run_repository = agent_run_repository
        self.tool_call_repository = tool_call_repository
        self.model_provider = normalized_provider
        self.model_name = normalized_model_name
        self.tool_versions = {
            contract.name: contract.version
            for contract in agent_runner.tool_contracts
        }

    def run(
        self,
        *,
        db: Session,
        context: ToolExecutionContext,
        message: str,
        observer: AgentRunObserver | None = None,
    ) -> NativeAgentResult:
        """执行并持久化一次同步 Agent Run。"""

        final_result: NativeAgentResult | None = None

        for event in self.run_events(
            db=db,
            context=context,
            message=message,
            observer=observer,
        ):
            if isinstance(event, AgentMessageEvent):
                final_result = NativeAgentResult(
                    answer=event.content,
                    turns=event.turns,
                    tool_call_count=event.tool_call_count,
                )

        if final_result is None:
            raise RuntimeError("agent run completed without final answer")

        return final_result

    def run_events(
        self,
        *,
        db: Session,
        context: ToolExecutionContext,
        message: str,
        observer: AgentRunObserver | None = None,
    ) -> Iterator[AgentRunEvent]:
        """执行 Agent，并把 Runtime 事件持久化为生命周期事实。"""

        normalized_message = self._normalize_message(message)
        agent_run = self._start_run(db=db, context=context)
        run_context = context.model_copy(
            update={"agent_run_id": agent_run.id}
        )
        event_stream: Iterator[AgentRunEvent] | None = None
        open_tool_calls: dict[str, int] = {}
        tool_call_count = 0
        completed = False

        try:
            event_stream = self.agent_runner.run_events(
                db=db,
                context=run_context,
                message=normalized_message,
                observer=observer,
            )

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
                    tool_call_id = open_tool_calls.pop(
                        event.call_id,
                        None,
                    )
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
                    "agent event stream completed without final answer"
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

    def _start_run(
        self,
        *,
        db: Session,
        context: ToolExecutionContext,
    ) -> AgentRun:
        agent_run = AgentRun(
            user_id=context.user_id,
            knowledge_base_id=context.knowledge_base_id,
            request_id=context.request_id,
            status=AgentRunStatus.RUNNING.value,
            model_provider=self.model_provider,
            model_name=self.model_name,
            tool_call_count=0,
            error_type=None,
        )

        try:
            self.agent_run_repository.create(db, agent_run)
            db.commit()
            db.refresh(agent_run)
            return agent_run
        except Exception:
            db.rollback()
            raise

    def _start_tool_call(
        self,
        *,
        db: Session,
        agent_run_id: int,
        event: AgentToolCallEvent,
    ) -> AgentToolCall:
        tool_call = AgentToolCall(
            agent_run_id=agent_run_id,
            provider_call_id=event.call_id,
            tool_name=event.tool_name,
            tool_version=self.tool_versions.get(
                event.tool_name,
                "unknown",
            ),
            status=AgentToolCallStatus.RUNNING.value,
            duration_ms=None,
            error_type=None,
        )

        try:
            self.tool_call_repository.create(db, tool_call)
            db.commit()
            db.refresh(tool_call)
            return tool_call
        except Exception:
            db.rollback()
            raise

    def _finish_tool_call(
        self,
        *,
        db: Session,
        tool_call_id: int,
        event: AgentToolResultEvent,
    ) -> None:
        tool_call = self._get_tool_call(
            db=db,
            tool_call_id=tool_call_id,
        )
        tool_call.status = (
            AgentToolCallStatus.SUCCEEDED.value
            if event.ok
            else AgentToolCallStatus.FAILED.value
        )
        tool_call.duration_ms = event.duration_ms
        tool_call.error_type = event.error_code
        tool_call.finished_at = datetime.now(timezone.utc)

        try:
            db.commit()
        except Exception:
            db.rollback()
            raise

    def _succeed_run(
        self,
        *,
        db: Session,
        agent_run_id: int,
        tool_call_count: int,
    ) -> None:
        agent_run = self._get_run(db=db, agent_run_id=agent_run_id)
        agent_run.status = AgentRunStatus.SUCCEEDED.value
        agent_run.tool_call_count = tool_call_count
        agent_run.error_type = None
        agent_run.finished_at = datetime.now(timezone.utc)

        try:
            db.commit()
        except Exception:
            db.rollback()
            raise

    def _fail_run(
        self,
        *,
        db: Session,
        agent_run_id: int,
        tool_call_count: int,
        error_type: str,
    ) -> None:
        db.rollback()
        agent_run = self._get_run(db=db, agent_run_id=agent_run_id)
        agent_run.status = AgentRunStatus.FAILED.value
        agent_run.tool_call_count = tool_call_count
        agent_run.error_type = self._normalize_error_type(error_type)
        agent_run.finished_at = datetime.now(timezone.utc)
        db.commit()

    def _fail_open_tool_calls(
        self,
        *,
        db: Session,
        open_tool_calls: dict[str, int],
        error_type: str,
    ) -> None:
        if not open_tool_calls:
            return

        db.rollback()
        normalized_error = self._normalize_error_type(error_type)

        for tool_call_id in open_tool_calls.values():
            tool_call = self._get_tool_call(
                db=db,
                tool_call_id=tool_call_id,
            )
            tool_call.status = AgentToolCallStatus.FAILED.value
            tool_call.error_type = normalized_error
            tool_call.finished_at = datetime.now(timezone.utc)

        db.commit()
        open_tool_calls.clear()

    def _safe_fail_run(self, **kwargs: Any) -> None:
        try:
            self._fail_run(**kwargs)
        except Exception as exc:
            logger.error(
                "Failed to persist agent run failure: error_type=%s",
                type(exc).__name__,
            )

    def _safe_fail_open_tool_calls(self, **kwargs: Any) -> None:
        try:
            self._fail_open_tool_calls(**kwargs)
        except Exception as exc:
            logger.error(
                "Failed to persist open tool call failure: error_type=%s",
                type(exc).__name__,
            )

    def _get_run(
        self,
        *,
        db: Session,
        agent_run_id: int,
    ) -> AgentRun:
        agent_run = self.agent_run_repository.find_by_id(
            db=db,
            agent_run_id=agent_run_id,
        )
        if agent_run is None:
            raise RuntimeError("agent run not found")
        return agent_run

    def _get_tool_call(
        self,
        *,
        db: Session,
        tool_call_id: int,
    ) -> AgentToolCall:
        tool_call = self.tool_call_repository.find_by_id(
            db=db,
            tool_call_id=tool_call_id,
        )
        if tool_call is None:
            raise RuntimeError("agent tool call not found")
        return tool_call

    @staticmethod
    def _normalize_message(message: str) -> str:
        if not message or not message.strip():
            raise ValueError("message cannot be empty")
        return message.strip()

    @staticmethod
    def _normalize_error_type(error_type: str) -> str:
        normalized = error_type.strip() or "unknown_error"
        return normalized[:100]

    @staticmethod
    def _close_iterator(iterator: Any | None) -> None:
        if iterator is None:
            return
        close_method = getattr(iterator, "close", None)
        if callable(close_method):
            close_method()
