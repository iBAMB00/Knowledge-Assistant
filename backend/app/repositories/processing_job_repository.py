from sqlalchemy.orm import Session

from app.constants.processing_job_status import (
    ProcessingJobStatus,
)
from app.models.database.processing_job import (
    ProcessingJob,
)


class ProcessingJobRepository:
    """
    文档处理任务仓库。

    只负责数据库读写和flush，
    不负责commit或rollback。
    """

    def create(
        self,
        db: Session,
        job: ProcessingJob,
    ) -> ProcessingJob:
        """
        创建任务记录。
        """

        db.add(job)
        db.flush()

        return job

    def find_by_id(
        self,
        db: Session,
        job_id: int,
    ) -> ProcessingJob | None:
        """
        根据ID查询任务。
        """

        return (
            db.query(ProcessingJob)
            .filter(
                ProcessingJob.id == job_id
            )
            .first()
        )

    def find_active_by_document_id(
        self,
        db: Session,
        document_id: int,
    ) -> ProcessingJob | None:
        """
        查询文档当前的活动任务。

        活动状态包括pending和running。
        """

        return (
            db.query(ProcessingJob)
            .filter(
                ProcessingJob.document_id
                == document_id,
                ProcessingJob.status.in_([
                    ProcessingJobStatus.PENDING.value,
                    ProcessingJobStatus.RUNNING.value,
                ]),
            )
            .order_by(
                ProcessingJob.id.desc()
            )
            .first()
        )

    def exists_active_by_document_id(
        self,
        db: Session,
        document_id: int,
    ) -> bool:
        """
        判断文档是否存在活动任务。

        只查询任务ID，避免为了冲突校验加载完整任务对象。
        """

        return (
            db.query(ProcessingJob.id)
            .filter(
                ProcessingJob.document_id == document_id,
                ProcessingJob.status.in_([
                    ProcessingJobStatus.PENDING.value,
                    ProcessingJobStatus.RUNNING.value,
                ]),
            )
            .first()
            is not None
        )

    def find_latest_by_document_id(
        self,
        db: Session,
        document_id: int,
    ) -> ProcessingJob | None:
        """
        查询文档最近一次任务。
        """

        return (
            db.query(ProcessingJob)
            .filter(
                ProcessingJob.document_id
                == document_id
            )
            .order_by(
                ProcessingJob.id.desc()
            )
            .first()
        )
    
    def find_all_by_document_id(
        self,
        db: Session,
        document_id: int,
    ) -> list[ProcessingJob]:
        """
        查询文档全部处理任务。

        按任务ID倒序排列，最新任务优先返回。
        """

        return (
            db.query(ProcessingJob)
            .filter(
                ProcessingJob.document_id
                == document_id
            )
            .order_by(
                ProcessingJob.id.desc()
            )
            .all()
        )