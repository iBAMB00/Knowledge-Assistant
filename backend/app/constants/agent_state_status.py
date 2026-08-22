from enum import Enum


class AgentStateStatus(str, Enum):
    """
    Stateful Agent 的状态机语义。

    它与 AgentRunStatus 不等价：AgentRun 记录一次执行事实，
    AgentState 描述可被 Checkpoint / Resume 延续的线程状态。
    """

    READY = "ready"
    RUNNING = "running"
    WAITING = "waiting"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
