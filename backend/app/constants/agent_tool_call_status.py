from enum import Enum


class AgentToolCallStatus(str, Enum):
    """持久化 ToolCall 当前最小生命周期状态。"""

    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
