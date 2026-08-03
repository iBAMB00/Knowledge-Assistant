import logging
from collections.abc import Callable

from sqlalchemy.orm import Session

from app.services.processing_job_executor import ProcessingJobExecutor


logger = logging.getLogger(__name__)


class ProcessingJobRunner:
    """
    使用独立数据库 Session 执行处理任务。

    Runner 只负责运行环境：
    - 创建数据库 Session
    - 调用 ProcessingJobExecutor
    - 处理未捕获异常
    - 关闭数据库 Session
    """

    def __init__(
        self,
        session_factory: Callable[[], Session],
        executor: ProcessingJobExecutor,
    ) -> None:
        self.session_factory = session_factory
        self.executor = executor

    def run(self, job_id: int) -> None:
        """
        根据任务 ID 执行后台处理任务。
        """

        db = self.session_factory()

        try:
            self.executor.execute_job(
                db=db,
                job_id=job_id,
            )

        except Exception:
            db.rollback()

            logger.exception(
                "processing job runner failed: job_id=%s",
                job_id,
            )

        finally:
            db.close()