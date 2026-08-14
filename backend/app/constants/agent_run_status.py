from enum import Enum


class AgentRunStatus(str, Enum):
    """AgentRun 当前最小生命周期状态。"""

    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
