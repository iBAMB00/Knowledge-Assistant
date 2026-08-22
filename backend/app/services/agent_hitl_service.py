"""v2.3-A8 durable Human-in-the-Loop approval service."""

from sqlalchemy.orm import Session

from app.agent.checkpoint import AgentExecutionCheckpointPayload
from app.agent.hitl import (
    AgentApprovalSelection,
    AgentApprovalStateError,
)
from app.constants.agent_state_status import AgentStateStatus
from app.services.agent_checkpoint_service import AgentCheckpointService


class AgentHITLService:
    """
    在可信 Conversation Scope 下对 WAITING checkpoint 做显式批准/拒绝。

    本服务只负责 Runtime interrupt lifecycle，不实现 v2.6 的角色审批矩阵、
    多级审批或风险策略。调用方身份已经通过 user_id / knowledge_base_id 进入
    AgentCheckpointService 的 ownership 隔离。
    """

    def __init__(
        self,
        checkpoint_service: AgentCheckpointService | None = None,
    ) -> None:
        self.checkpoint_service = (
            checkpoint_service or AgentCheckpointService()
        )

    def load_waiting_checkpoint(
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
            raise AgentApprovalStateError(
                "waiting checkpoint not found"
            )
        if payload.agent_state.status is not AgentStateStatus.WAITING:
            raise AgentApprovalStateError(
                "agent thread is not waiting for approval"
            )
        if not payload.pending_approvals:
            raise AgentApprovalStateError(
                "waiting checkpoint has no pending approvals"
            )
        if not payload.pending_tool_calls:
            raise AgentApprovalStateError(
                "waiting checkpoint has no pending tool calls"
            )
        return payload

    def approve(
        self,
        db: Session,
        *,
        thread_id: str,
        user_id: int,
        knowledge_base_id: int,
        selection: AgentApprovalSelection | None = None,
    ) -> AgentExecutionCheckpointPayload:
        payload = self.load_waiting_checkpoint(
            db,
            thread_id=thread_id,
            user_id=user_id,
            knowledge_base_id=knowledge_base_id,
        )
        required_ids = {
            item.call_id for item in payload.pending_approvals
        }
        selected_ids = (
            set(selection.call_ids)
            if selection is not None and selection.call_ids
            else required_ids
        )
        unknown = selected_ids - required_ids
        if unknown:
            raise AgentApprovalStateError(
                "approval contains unknown call_id"
            )
        if set(payload.rejected_call_ids) & selected_ids:
            raise AgentApprovalStateError(
                "rejected tool call cannot be approved"
            )

        approved_ids = tuple(
            dict.fromkeys(
                (*payload.approved_call_ids, *selected_ids)
            )
        )
        updated = AgentExecutionCheckpointPayload.model_validate(
            {
                **payload.model_dump(mode="python"),
                "approved_call_ids": approved_ids,
            }
        )
        # 仍保存 WAITING：只有真正启动 resume_after_approval 时，Runner 才把
        # Thread 变为 RUNNING，避免“已批准但尚未续跑”被误判成正在执行。
        self.checkpoint_service.save_checkpoint(db, updated)
        return updated

    def reject(
        self,
        db: Session,
        *,
        thread_id: str,
        user_id: int,
        knowledge_base_id: int,
        selection: AgentApprovalSelection | None = None,
    ) -> AgentExecutionCheckpointPayload:
        payload = self.load_waiting_checkpoint(
            db,
            thread_id=thread_id,
            user_id=user_id,
            knowledge_base_id=knowledge_base_id,
        )
        required_ids = {
            item.call_id for item in payload.pending_approvals
        }
        selected_ids = (
            set(selection.call_ids)
            if selection is not None and selection.call_ids
            else required_ids
        )
        unknown = selected_ids - required_ids
        if unknown:
            raise AgentApprovalStateError(
                "rejection contains unknown call_id"
            )
        if set(payload.approved_call_ids) & selected_ids:
            raise AgentApprovalStateError(
                "approved tool call cannot be rejected"
            )

        rejected_ids = tuple(
            dict.fromkeys(
                (*payload.rejected_call_ids, *selected_ids)
            )
        )
        cancelled_state = payload.agent_state.model_copy(
            update={
                "status": AgentStateStatus.CANCELLED,
                "last_error_code": "approval_rejected",
            }
        )
        updated = AgentExecutionCheckpointPayload.model_validate(
            {
                **payload.model_dump(mode="python"),
                "agent_state": cancelled_state,
                "rejected_call_ids": rejected_ids,
            }
        )
        self.checkpoint_service.save_checkpoint(db, updated)
        return updated

    def load_approved_checkpoint(
        self,
        db: Session,
        *,
        thread_id: str,
        user_id: int,
        knowledge_base_id: int,
    ) -> AgentExecutionCheckpointPayload:
        payload = self.load_waiting_checkpoint(
            db,
            thread_id=thread_id,
            user_id=user_id,
            knowledge_base_id=knowledge_base_id,
        )
        required_ids = {
            item.call_id for item in payload.pending_approvals
        }
        approved_ids = set(payload.approved_call_ids)
        rejected_ids = set(payload.rejected_call_ids)

        if rejected_ids:
            raise AgentApprovalStateError(
                "waiting checkpoint contains rejected tool calls"
            )
        if not required_ids.issubset(approved_ids):
            raise AgentApprovalStateError(
                "not all pending tool calls are approved"
            )
        return payload
