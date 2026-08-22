"""Framework-neutral Stateful Agent cancellation/run-control contracts."""

from typing import Protocol

from sqlalchemy.orm import Session


class AgentRunControlError(RuntimeError):
    """Stateful Agent 运行控制失败。"""


class AgentRunControlNotFoundError(AgentRunControlError):
    """当前可信 Scope 下找不到 Thread/checkpoint。"""


class AgentRunStateError(AgentRunControlError):
    """Thread 当前生命周期状态不允许执行请求的控制操作。"""


class AgentRunCancellationError(AgentRunControlError):
    """执行已被显式取消，不允许继续模型或 Tool 执行。"""


class AgentCancellationProbe(Protocol):
    """LangGraph Runner 依赖的最小 cooperative cancellation 边界。"""

    def raise_if_cancelled(
        self,
        db: Session,
        *,
        thread_id: str,
        user_id: int,
        knowledge_base_id: int,
    ) -> None:
        ...
