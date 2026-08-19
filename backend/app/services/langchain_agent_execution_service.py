"""LangChain Candidate 到 AgentRun / AgentToolCall 的生命周期持久化桥。"""

import logging
from datetime import datetime, timezone
from collections.abc import Iterator
from typing import Any

from sqlalchemy.orm import Session

from app.agent.context import ToolExecutionContext
from app.agent.frameworks.langchain.execution_observer import (
    LangChainToolExecutionObserver,
)
from app.agent.frameworks.langchain.runner import (
    LangChainAgentError,
    LangChainAgentResult,
    LangChainSingleAgentRunner,
)
from app.agent.run_event import AgentMessageEvent, AgentRunEvent
from app.agent.run_observer import AgentRunObserver
from app.agent.version_snapshot import (
    AgentEvaluationVersionContext,
    AgentRuntimeVersionSnapshot,
)
from app.constants.agent_run_status import AgentRunStatus
from app.constants.agent_tool_call_status import AgentToolCallStatus
from app.models.database.agent_run import AgentRun
from app.models.database.agent_tool_call import AgentToolCall
from app.repositories.agent_run_repository import AgentRunRepository
from app.repositories.agent_tool_call_repository import AgentToolCallRepository


logger = logging.getLogger(__name__)


class _PersistedToolExecutionObserver(LangChainToolExecutionObserver):
    """把真正执行的 LangChain Tool 写成 AgentToolCall 生命周期事实。"""

    def __init__(
        self,
        *,
        service: "LangChainAgentExecutionService",
        db: Session,
        agent_run_id: int,
    ) -> None:
        self._service = service
        self._db = db
        self._agent_run_id = agent_run_id
        self._open_tool_calls: dict[str, int] = {}
        self.executed_tool_call_count = 0

    def on_tool_execution_started(
        self,
        *,
        call_id: str,
        tool_name: str,
    ) -> None:
        if call_id in self._open_tool_calls:
            raise RuntimeError("duplicate open langchain tool call")

        tool_call = self._service._start_tool_call(
            db=self._db,
            agent_run_id=self._agent_run_id,
            call_id=call_id,
            tool_name=tool_name,
        )
        self._open_tool_calls[call_id] = tool_call.id
        self.executed_tool_call_count += 1

    def on_tool_execution_finished(
        self,
        *,
        call_id: str,
        tool_name: str,
        ok: bool,
        error_code: str | None,
        duration_ms: int,
    ) -> None:
        tool_call_id = self._open_tool_calls.get(call_id)
        if tool_call_id is None:
            raise RuntimeError("langchain tool result has no persisted tool call")

        self._service._finish_tool_call(
            db=self._db,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            ok=ok,
            error_code=error_code,
            duration_ms=duration_ms,
        )
        self._open_tool_calls.pop(call_id, None)

    def fail_open(self, *, error_type: str) -> None:
        self._service._safe_fail_open_tool_calls(
            db=self._db,
            open_tool_calls=self._open_tool_calls,
            error_type=error_type,
        )


class LangChainAgentExecutionService:
    """
    将 LangChain Candidate Runtime 映射为现有 Agent 生命周期事实。

    与 Native AgentExecutionService 保持相同的数据安全边界：
    - AgentRun 保存身份、Scope、状态与版本快照；
    - AgentToolCall 只保存 Tool 名称/版本/状态/耗时/安全错误码；
    - 不保存 Prompt、模型回答、Tool 参数正文、Tool Result 正文或 CoT。
    """

    def __init__(
        self,
        *,
        agent_runner: LangChainSingleAgentRunner,
        agent_run_repository: AgentRunRepository,
        tool_call_repository: AgentToolCallRepository,
        model_provider: str,
        model_name: str,
        version_snapshot: AgentRuntimeVersionSnapshot,
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
        self.version_snapshot = version_snapshot
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
        evaluation_version: AgentEvaluationVersionContext | None = None,
    ) -> LangChainAgentResult:
        """执行并持久化一次同步 Candidate Run。"""

        final_result: LangChainAgentResult | None = None
        for event in self.run_events(
            db=db,
            context=context,
            message=message,
            observer=observer,
            evaluation_version=evaluation_version,
        ):
            if isinstance(event, AgentMessageEvent):
                final_result = LangChainAgentResult(
                    answer=event.content,
                    turns=event.turns,
                    tool_call_count=event.tool_call_count,
                )

        if final_result is None:
            raise RuntimeError(
                "langchain agent run completed without final answer"
            )
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
        """执行 Candidate 事件流，并持久化 AgentRun / AgentToolCall。"""

        normalized_message = self._normalize_message(message)
        agent_run = self._start_run(
            db=db,
            context=context,
            evaluation_version=evaluation_version,
        )
        run_context = context.model_copy(
            update={"agent_run_id": agent_run.id}
        )
        execution_observer = _PersistedToolExecutionObserver(
            service=self,
            db=db,
            agent_run_id=agent_run.id,
        )
        event_stream: Iterator[AgentRunEvent] | None = None
        completed = False

        try:
            event_stream = self.agent_runner.run_events(
                db=db,
                context=run_context,
                message=normalized_message,
                observer=observer,
                execution_observer=execution_observer,
            )

            for event in event_stream:
                if isinstance(event, AgentMessageEvent):
                    self._succeed_run(
                        db=db,
                        agent_run_id=agent_run.id,
                        tool_call_count=(
                            execution_observer.executed_tool_call_count
                        ),
                    )
                    completed = True
                yield event

            if not completed:
                raise RuntimeError(
                    "langchain agent event stream completed without final answer"
                )

        except GeneratorExit:
            if not completed:
                execution_observer.fail_open(
                    error_type="stream_cancelled"
                )
                self._safe_fail_run(
                    db=db,
                    agent_run_id=agent_run.id,
                    tool_call_count=(
                        execution_observer.executed_tool_call_count
                    ),
                    error_type="stream_cancelled",
                )
            raise

        except LangChainAgentError as exc:
            execution_observer.fail_open(error_type=exc.code)
            self._safe_fail_run(
                db=db,
                agent_run_id=agent_run.id,
                tool_call_count=execution_observer.executed_tool_call_count,
                error_type=exc.code,
            )
            raise

        except Exception as exc:
            error_type = type(exc).__name__
            execution_observer.fail_open(error_type=error_type)
            self._safe_fail_run(
                db=db,
                agent_run_id=agent_run.id,
                tool_call_count=execution_observer.executed_tool_call_count,
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
        evaluation_version: AgentEvaluationVersionContext | None,
    ) -> AgentRun:
        agent_run = AgentRun(
            user_id=context.user_id,
            knowledge_base_id=context.knowledge_base_id,
            request_id=context.request_id,
            status=AgentRunStatus.RUNNING.value,
            model_provider=self.model_provider,
            model_name=self.model_name,
            agent_version=self.version_snapshot.agent_version,
            prompt_version=self.version_snapshot.prompt_version,
            toolset_version=self.version_snapshot.toolset_version,
            retrieval_config_version=(
                self.version_snapshot.retrieval_config_version
            ),
            eval_dataset_version=(
                evaluation_version.dataset_version
                if evaluation_version is not None
                else None
            ),
            evaluator_version=(
                evaluation_version.evaluator_version
                if evaluation_version is not None
                else None
            ),
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
        call_id: str,
        tool_name: str,
    ) -> AgentToolCall:
        tool_call = AgentToolCall(
            agent_run_id=agent_run_id,
            provider_call_id=call_id,
            tool_name=tool_name,
            tool_version=self.tool_versions.get(tool_name, "unknown"),
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
        tool_name: str,
        ok: bool,
        error_code: str | None,
        duration_ms: int,
    ) -> None:
        tool_call = self._get_tool_call(db=db, tool_call_id=tool_call_id)
        if tool_call.tool_name != tool_name:
            raise RuntimeError("langchain tool result name mismatch")
        tool_call.status = (
            AgentToolCallStatus.SUCCEEDED.value
            if ok
            else AgentToolCallStatus.FAILED.value
        )
        tool_call.duration_ms = max(0, duration_ms)
        tool_call.error_type = None if ok else self._normalize_error_type(
            error_code or "tool_error"
        )
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
            tool_call = self._get_tool_call(db=db, tool_call_id=tool_call_id)
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
                "Failed to persist langchain agent run failure: error_type=%s",
                type(exc).__name__,
            )

    def _safe_fail_open_tool_calls(self, **kwargs: Any) -> None:
        try:
            self._fail_open_tool_calls(**kwargs)
        except Exception as exc:
            logger.error(
                "Failed to persist langchain open tool call failure: error_type=%s",
                type(exc).__name__,
            )

    def _get_run(self, *, db: Session, agent_run_id: int) -> AgentRun:
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
    def _close_iterator(iterator: Any | None) -> None:
        """客户端取消或异常时尽量关闭底层 Candidate 事件生成器。"""

        if iterator is None:
            return
        close_method = getattr(iterator, "close", None)
        if not callable(close_method):
            return
        try:
            close_method()
        except Exception as exc:
            logger.warning(
                "Failed to close langchain agent event stream: error_type=%s",
                type(exc).__name__,
            )

    @staticmethod
    def _normalize_message(message: str) -> str:
        if not message or not message.strip():
            raise ValueError("message cannot be empty")
        return message.strip()

    @staticmethod
    def _normalize_error_type(error_type: str) -> str:
        normalized = error_type.strip() or "unknown_error"
        return normalized[:100]
