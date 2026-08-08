from typing import Protocol


class CeleryTaskLike(Protocol):
    """描述任务派发所需的最小 Celery Task 接口。"""

    def delay(self, job_id: int): ...


class ProcessingJobDispatchError(RuntimeError):
    """ProcessingJob 无法提交到消息队列。"""


class ProcessingJobDispatcher:
    """负责把已经持久化的 ProcessingJob 派发到 Celery。"""

    def __init__(self, task: CeleryTaskLike) -> None:
        self.task = task

    def dispatch(self, job_id: int) -> None:
        """仅把 job_id 放入消息队列，不传递数据库对象或文件对象。"""
        try:
            self.task.delay(job_id)
        except Exception as exc:
            raise ProcessingJobDispatchError(
                "processing job dispatch failed"
            ) from exc
