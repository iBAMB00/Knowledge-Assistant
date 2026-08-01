from enum import Enum


class ProcessingJobStatus(str, Enum):
    """
    文档处理任务状态。
    """

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"