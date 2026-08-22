"""v2.3-A10 safe query boundary for Stateful Agent thread status."""

from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.agent.checkpoint import AgentExecutionCheckpointPayload
from app.constants.agent_state_status import AgentStateStatus
from app.services.agent_checkpoint_service import AgentCheckpointService


class AgentThreadNotFoundError(RuntimeError):
    """当前可信 Scope 下不存在 Thread。"""


@dataclass(frozen=True)
class AgentThreadStatusSnapshot:
    payload: AgentExecutionCheckpointPayload
    can_resume: bool
    can_approve: bool
    can_reject: bool
    can_cancel: bool


class AgentThreadStatusService:
    """从最新 durable checkpoint 派生对外最小 Thread 控制能力。"""

    def __init__(
        self,
        checkpoint_service: AgentCheckpointService | None = None,
    ) -> None:
        self.checkpoint_service = (
            checkpoint_service or AgentCheckpointService()
        )

    def get_status(
        self,
        db: Session,
        *,
        thread_id: str,
        user_id: int,
        knowledge_base_id: int,
    ) -> AgentThreadStatusSnapshot:
        payload = self.checkpoint_service.load_latest_for_scope(
            db,
            thread_id=thread_id,
            user_id=user_id,
            knowledge_base_id=knowledge_base_id,
        )
        if payload is None:
            raise AgentThreadNotFoundError("agent thread not found")

        status = payload.agent_state.status
        required_ids = {
            item.call_id for item in payload.pending_approvals
        }
        approved_ids = set(payload.approved_call_ids)
        rejected_ids = set(payload.rejected_call_ids)
        unresolved_ids = required_ids - approved_ids - rejected_ids

        waiting = status is AgentStateStatus.WAITING
        resume_after_approval = (
            waiting
            and bool(required_ids)
            and not rejected_ids
            and required_ids.issubset(approved_ids)
        )

        return AgentThreadStatusSnapshot(
            payload=payload,
            can_resume=(
                status is AgentStateStatus.RUNNING
                or resume_after_approval
            ),
            can_approve=waiting and bool(unresolved_ids),
            can_reject=waiting and bool(unresolved_ids),
            can_cancel=status in {
                AgentStateStatus.RUNNING,
                AgentStateStatus.WAITING,
            },
        )
