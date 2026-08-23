"""Framework-neutral durable checkpoint, recovery and HITL contracts."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy.orm import Session

from app.agent.hitl import AgentApprovalRequirement
from app.agent.model_response import (
    LLMToolCall,
    LLMToolExchange,
    LLMToolResponse,
)
from app.agent.run_event import AgentToolResultEvent
from app.agent.state import AGENT_STATE_SCHEMA_VERSION, AgentState
from app.constants.agent_state_status import AgentStateStatus


CHECKPOINT_SCHEMA_VERSION = "1.1"
SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS = {"1.0", "1.1"}


class AgentCheckpointStateTransitionError(RuntimeError):
    """持久化 Thread 的当前状态不允许写入请求的下一状态。"""

    def __init__(
        self,
        *,
        current_status: AgentStateStatus,
        requested_status: AgentStateStatus,
        current_agent_run_id: int | str | None = None,
        requested_agent_run_id: int | str | None = None,
    ) -> None:
        self.current_status = current_status
        self.requested_status = requested_status
        self.current_agent_run_id = current_agent_run_id
        self.requested_agent_run_id = requested_agent_run_id
        super().__init__(
            "invalid checkpoint state transition: "
            f"{current_status.value} -> {requested_status.value}"
        )


class AgentExecutionCheckpointPayload(BaseModel):
    """可序列化、可落库的 Graph 执行快照。"""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    checkpoint_schema_version: str = CHECKPOINT_SCHEMA_VERSION
    agent_state: AgentState
    history: tuple[LLMToolExchange, ...] = ()
    pending_tool_calls: tuple[LLMToolCall, ...] = ()
    last_model_response: LLMToolResponse | None = None
    tool_observations: tuple[AgentToolResultEvent, ...] = ()
    final_answer: str | None = None
    turn: int = Field(default=0, ge=0)
    tool_call_count: int = Field(default=0, ge=0)
    seen_tool_call_signatures: tuple[str, ...] = ()

    # v2.3-A8：只保存安全审批元数据和 call id 决策，不新增隐藏推理。
    pending_approvals: tuple[AgentApprovalRequirement, ...] = ()
    approved_call_ids: tuple[str, ...] = ()
    rejected_call_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_checkpoint_contract(self) -> "AgentExecutionCheckpointPayload":
        if (
            self.checkpoint_schema_version
            not in SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS
        ):
            raise ValueError("unsupported checkpoint schema version")
        if self.agent_state.state_schema_version != AGENT_STATE_SCHEMA_VERSION:
            raise ValueError("unsupported agent state schema version")

        if self.pending_tool_calls:
            if self.last_model_response is None:
                raise ValueError(
                    "pending tool calls require last model response"
                )
            if tuple(self.last_model_response.tool_calls) != self.pending_tool_calls:
                raise ValueError(
                    "pending tool calls do not match last model response"
                )

        pending_call_ids = {
            tool_call.id for tool_call in self.pending_tool_calls
        }
        approval_call_ids = {
            item.call_id for item in self.pending_approvals
        }
        if not approval_call_ids.issubset(pending_call_ids):
            raise ValueError(
                "pending approvals must reference pending tool calls"
            )

        approved_ids = tuple(
            call_id.strip() for call_id in self.approved_call_ids
        )
        rejected_ids = tuple(
            call_id.strip() for call_id in self.rejected_call_ids
        )
        if any(not call_id for call_id in (*approved_ids, *rejected_ids)):
            raise ValueError("approval decision call_id cannot be empty")
        if len(set(approved_ids)) != len(approved_ids):
            raise ValueError("approved call_ids must be unique")
        if len(set(rejected_ids)) != len(rejected_ids):
            raise ValueError("rejected call_ids must be unique")
        if set(approved_ids) & set(rejected_ids):
            raise ValueError("approval decision cannot be both approved and rejected")
        if not set(approved_ids).issubset(approval_call_ids):
            raise ValueError("approved call_ids require pending approval")
        if not set(rejected_ids).issubset(approval_call_ids):
            raise ValueError("rejected call_ids require pending approval")

        if self.agent_state.status is AgentStateStatus.WAITING:
            if not self.pending_approvals:
                raise ValueError("waiting state requires pending approvals")
            unresolved = approval_call_ids - set(approved_ids) - set(rejected_ids)
            if not unresolved and not rejected_ids:
                # 已全部批准的 checkpoint 仍保持 WAITING，直到 Runner 正式
                # 开始新的执行尝试；这是 durable approval 与 execution 的边界。
                pass
        elif self.pending_approvals and self.pending_tool_calls:
            # RUNNING checkpoint 可以在批准后恢复尝试的起始边界继续携带审批
            # 记录；其他终态不应遗留未消费的 approval state。
            if self.agent_state.status not in (
                AgentStateStatus.RUNNING,
                AgentStateStatus.CANCELLED,
            ):
                raise ValueError(
                    "pending approvals require waiting/running/cancelled state"
                )

        if self.final_answer is not None:
            if self.agent_state.status is not AgentStateStatus.SUCCEEDED:
                raise ValueError(
                    "final answer requires succeeded agent state"
                )
            if not self.final_answer.strip():
                raise ValueError("final answer cannot be empty")

        return self


class AgentCheckpointWriter(Protocol):
    """Runner 只依赖这一最小写入边界，不依赖 SQLAlchemy Repository。"""

    def save_checkpoint(
        self,
        db: Session,
        payload: AgentExecutionCheckpointPayload,
        *,
        allowed_previous_statuses: set[AgentStateStatus] | None = None,
        new_execution_attempt: bool = False,
    ) -> object:
        ...


class AgentRecoveryLoader(Protocol):
    """Runner 读取可恢复 RUNNING checkpoint 的最小边界。"""

    def load_resume_checkpoint(
        self,
        db: Session,
        *,
        thread_id: str,
        user_id: int,
        knowledge_base_id: int,
    ) -> AgentExecutionCheckpointPayload:
        ...


class AgentResumeError(RuntimeError):
    """Stateful Agent 恢复执行失败。"""


class AgentResumeCheckpointNotFoundError(AgentResumeError):
    """当前可信 Scope 下不存在可读取的 checkpoint。"""


class AgentResumeStateError(AgentResumeError):
    """Checkpoint 存在，但当前状态不允许恢复执行。"""
