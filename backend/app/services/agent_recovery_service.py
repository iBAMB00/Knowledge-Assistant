"""v2.3-A7 Stateful Agent recovery policy."""

from sqlalchemy.orm import Session

from app.agent.checkpoint import (
    AgentExecutionCheckpointPayload,
    AgentResumeCheckpointNotFoundError,
    AgentResumeStateError,
)
from app.constants.agent_state_status import AgentStateStatus
from app.services.agent_checkpoint_service import AgentCheckpointService


class AgentRecoveryService:
    """
    从 durable checkpoint 解析一次可恢复执行候选。

    Repository / DB Scope 校验仍由 AgentCheckpointService 负责；这里仅定义
    v2.3-A7 的恢复策略：只恢复因中断遗留的 RUNNING Thread。WAITING 属于
    后续 HITL，FAILED / CANCELLED / SUCCEEDED 都不在本版本隐式重跑。
    """

    def __init__(
        self,
        checkpoint_service: AgentCheckpointService | None = None,
    ) -> None:
        self.checkpoint_service = (
            checkpoint_service or AgentCheckpointService()
        )

    def load_resume_checkpoint(
        self,
        db: Session,
        *,
        thread_id: str,
        user_id: int,
        knowledge_base_id: int,
    ) -> AgentExecutionCheckpointPayload:
        payload = self.checkpoint_service.load_latest_for_scope(
            db,
            thread_id=thread_id,
            user_id=user_id,
            knowledge_base_id=knowledge_base_id,
        )
        if payload is None:
            raise AgentResumeCheckpointNotFoundError(
                "resume checkpoint not found"
            )

        state = payload.agent_state
        if state.thread.thread_id != thread_id:
            raise AgentResumeStateError(
                "resume checkpoint thread does not match request"
            )
        if state.status is not AgentStateStatus.RUNNING:
            raise AgentResumeStateError(
                f"agent thread is not resumable from status: "
                f"{state.status.value}"
            )
        if not state.task:
            raise AgentResumeStateError(
                "resume checkpoint is missing task"
            )
        if payload.final_answer is not None:
            raise AgentResumeStateError(
                "running checkpoint cannot contain final answer"
            )

        return payload
