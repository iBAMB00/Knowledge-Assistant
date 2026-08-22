"""v2.3-A9 durable Stateful Agent cancellation service."""

from sqlalchemy.orm import Session

from app.agent.checkpoint import (
    AgentCheckpointStateTransitionError,
    AgentExecutionCheckpointPayload,
)
from app.agent.run_control import (
    AgentRunCancellationError,
    AgentRunControlNotFoundError,
    AgentRunStateError,
)
from app.constants.agent_state_status import AgentStateStatus
from app.services.agent_checkpoint_service import AgentCheckpointService


class AgentRunControlService:
    """
    在可信 User / KB Scope 下提供 Thread 取消与 cooperative cancellation probe。

    A9 只负责 Runtime lifecycle：RUNNING / WAITING -> CANCELLED。
    谁可以取消、审计策略、后台超时扫描等治理能力留给后续版本。
    """

    CANCELLATION_ERROR_CODE = "cancelled_by_user"
    _CANCELLABLE_STATUSES = {
        AgentStateStatus.RUNNING,
        AgentStateStatus.WAITING,
    }

    def __init__(
        self,
        checkpoint_service: AgentCheckpointService | None = None,
    ) -> None:
        self.checkpoint_service = (
            checkpoint_service or AgentCheckpointService()
        )

    def cancel(
        self,
        db: Session,
        *,
        thread_id: str,
        user_id: int,
        knowledge_base_id: int,
    ) -> AgentExecutionCheckpointPayload:
        """显式取消 RUNNING / WAITING Thread，并持久化 CANCELLED checkpoint。"""

        normalized_thread_id = thread_id.strip()
        if not normalized_thread_id:
            raise AgentRunStateError("thread_id cannot be empty")

        payload = self.checkpoint_service.load_latest_for_scope(
            db,
            thread_id=normalized_thread_id,
            user_id=user_id,
            knowledge_base_id=knowledge_base_id,
        )
        if payload is None:
            raise AgentRunControlNotFoundError(
                "agent thread not found"
            )

        current_status = payload.agent_state.status
        if current_status is AgentStateStatus.CANCELLED:
            # Cancel 是幂等控制操作：重复请求不制造额外 checkpoint。
            return payload
        if current_status not in self._CANCELLABLE_STATUSES:
            raise AgentRunStateError(
                "agent thread is not cancellable from status: "
                f"{current_status.value}"
            )

        cancelled_state = payload.agent_state.model_copy(
            update={
                "status": AgentStateStatus.CANCELLED,
                "last_error_code": self.CANCELLATION_ERROR_CODE,
            }
        )
        cancelled_payload = AgentExecutionCheckpointPayload.model_validate(
            {
                **payload.model_dump(mode="python"),
                "agent_state": cancelled_state,
            }
        )

        try:
            self.checkpoint_service.save_checkpoint(
                db,
                cancelled_payload,
                allowed_previous_statuses=self._CANCELLABLE_STATUSES,
            )
        except AgentCheckpointStateTransitionError as exc:
            # 并发 Cancel：另一请求若已经先提交 CANCELLED，则保持幂等。
            latest = self.checkpoint_service.load_latest_for_scope(
                db,
                thread_id=normalized_thread_id,
                user_id=user_id,
                knowledge_base_id=knowledge_base_id,
            )
            if (
                latest is not None
                and latest.agent_state.status
                is AgentStateStatus.CANCELLED
            ):
                return latest
            raise AgentRunStateError(
                "agent thread changed state before cancellation completed"
            ) from exc

        return cancelled_payload

    def raise_if_cancelled(
        self,
        db: Session,
        *,
        thread_id: str,
        user_id: int,
        knowledge_base_id: int,
    ) -> None:
        """Runner 在模型/Tool 边界调用，发现 durable CANCELLED 即停止。"""

        payload = self.checkpoint_service.load_latest_for_scope(
            db,
            thread_id=thread_id,
            user_id=user_id,
            knowledge_base_id=knowledge_base_id,
        )
        if payload is None:
            return
        if payload.agent_state.status is AgentStateStatus.CANCELLED:
            raise AgentRunCancellationError(
                "agent execution was cancelled"
            )
