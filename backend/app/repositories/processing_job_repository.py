from collections.abc import Sequence

from sqlalchemy import select
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

    def find_active_by_document_ids(
        self,
        db: Session,
        document_ids: Sequence[int],
    ) -> dict[int, ProcessingJob]:
        """
        批量查询多份文档当前的活动任务。

        使用一次IN查询加载全部pending或running任务，
        返回以document_id为键的任务映射。
        """

        normalized_document_ids = sorted(set(document_ids))

        if not normalized_document_ids:
            return {}

        statement = (
            select(ProcessingJob)
            .where(
                ProcessingJob.document_id.in_(
                    normalized_document_ids
                ),
                ProcessingJob.status.in_([
                    ProcessingJobStatus.PENDING.value,
                    ProcessingJobStatus.RUNNING.value,
                ]),
            )
            .order_by(
                ProcessingJob.document_id.asc(),
                ProcessingJob.id.desc(),
            )
        )

        jobs = db.scalars(statement).all()

        active_jobs: dict[int, ProcessingJob] = {}

        for job in jobs:
            # 当前数据库已有“同一文档最多一个活动任务”的唯一索引。
            # setdefault同时兼容可能遗留的异常历史数据，
            # 保留ID最大的最新活动任务。
            active_jobs.setdefault(
                job.document_id,
                job,
            )

        return active_jobs

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