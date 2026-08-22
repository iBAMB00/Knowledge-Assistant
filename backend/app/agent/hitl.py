"""Framework-neutral Human-in-the-Loop contracts for Stateful Agent runtime."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy.orm import Session

from app.agent.model_response import LLMToolCall
from app.agent.tools.base import ToolContract

if TYPE_CHECKING:
    from app.agent.checkpoint import AgentExecutionCheckpointPayload


class AgentApprovalRequirement(BaseModel):
    """一次等待人工确认的安全元数据，不暴露 Tool 参数正文。"""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    call_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    reason: str = Field(min_length=1, max_length=500)


class AgentInterruptPolicy(Protocol):
    """决定某个 ToolCall 是否需要人工确认的最小策略边界。"""

    def get_approval_requirement(
        self,
        *,
        tool_call: LLMToolCall,
        tool_contract: ToolContract,
    ) -> AgentApprovalRequirement | None:
        ...


class NoAgentInterruptPolicy:
    """默认策略：当前现有 READ_ONLY Tool 不触发 HITL。"""

    def get_approval_requirement(
        self,
        *,
        tool_call: LLMToolCall,
        tool_contract: ToolContract,
    ) -> AgentApprovalRequirement | None:
        return None


class ToolNameApprovalPolicy:
    """
    最小可配置策略：按 Tool 名称要求人工确认。

    v2.3-A8 只验证 Runtime Interrupt/Resume 能力，不在这里实现 v2.6 的
    风险分级、角色审批矩阵或安全治理。生产侧后续可替换此 Protocol。
    """

    def __init__(self, requirements: Mapping[str, str]) -> None:
        normalized: dict[str, str] = {}
        for tool_name, reason in requirements.items():
            normalized_name = tool_name.strip()
            normalized_reason = reason.strip()
            if not normalized_name:
                raise ValueError("approval tool name cannot be empty")
            if not normalized_reason:
                raise ValueError("approval reason cannot be empty")
            normalized[normalized_name] = normalized_reason
        self._requirements = normalized

    def get_approval_requirement(
        self,
        *,
        tool_call: LLMToolCall,
        tool_contract: ToolContract,
    ) -> AgentApprovalRequirement | None:
        reason = self._requirements.get(tool_contract.name)
        if reason is None:
            return None
        if tool_call.name != tool_contract.name:
            raise ValueError("tool call does not match tool contract")
        return AgentApprovalRequirement(
            call_id=tool_call.id,
            tool_name=tool_call.name,
            reason=reason,
        )


class AgentHITLLoader(Protocol):
    """Runner 从 WAITING checkpoint 读取已批准状态的最小边界。"""

    def load_approved_checkpoint(
        self,
        db: Session,
        *,
        thread_id: str,
        user_id: int,
        knowledge_base_id: int,
    ) -> "AgentExecutionCheckpointPayload":
        ...


class AgentInterruptRequired(RuntimeError):
    """Graph 已安全停在 WAITING checkpoint，等待显式人工确认。"""

    def __init__(
        self,
        *,
        thread_id: str,
        approvals: Sequence[AgentApprovalRequirement],
    ) -> None:
        self.thread_id = thread_id
        self.approvals = tuple(approvals)
        tool_names = ", ".join(item.tool_name for item in self.approvals)
        super().__init__(
            f"agent execution is waiting for approval: {tool_names}"
        )


class AgentApprovalStateError(RuntimeError):
    """当前 Thread / Checkpoint 不满足批准、拒绝或批准后续跑条件。"""


class AgentApprovalSelection(BaseModel):
    """对待确认 call id 的显式选择，避免静默批准未知 ToolCall。"""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    call_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_unique_call_ids(self) -> "AgentApprovalSelection":
        normalized = tuple(call_id.strip() for call_id in self.call_ids)
        if any(not call_id for call_id in normalized):
            raise ValueError("approval call_id cannot be empty")
        if len(set(normalized)) != len(normalized):
            raise ValueError("approval call_ids must be unique")
        object.__setattr__(self, "call_ids", normalized)
        return self
