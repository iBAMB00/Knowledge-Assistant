from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)

from app.constants.processing_job_status import (
    ProcessingJobStatus,
)
from app.core.database import Base


class ProcessingJob(Base):
    """
    文档处理任务记录。

    一个任务代表一次处理尝试，用于记录：
    - 任务类型
    - 执行状态
    - 当前进度
    - 失败原因
    - 开始和结束时间

    当前模型只负责持久化，不负责执行任务。
    """

    __tablename__ = "processing_jobs"

    __table_args__ = (
        CheckConstraint(
            (
                "job_type IN "
                "('document_processing', 'embedding')"
            ),
            name="ck_processing_jobs_job_type",
        ),
        CheckConstraint(
            (
                "status IN "
                "('pending', 'running', "
                "'succeeded', 'failed')"
            ),
            name="ck_processing_jobs_status",
        ),
        CheckConstraint(
            "progress >= 0 AND progress <= 100",
            name="ck_processing_jobs_progress",
        ),
        Index(
            "ix_processing_jobs_document_status",
            "document_id",
            "status",
        ),
    )

    id = Column(
        Integer,
        primary_key=True,
        index=True,
    )

    document_id = Column(
        Integer,
        ForeignKey(
            "documents.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    job_type = Column(
        String(32),
        nullable=False,
    )

    status = Column(
        String(20),
        nullable=False,
        default=ProcessingJobStatus.PENDING.value,
    )

    progress = Column(
        Integer,
        nullable=False,
        default=0,
    )

    error_message = Column(
        Text,
        nullable=True,
    )

    created_at = Column(
        DateTime,
        nullable=False,
        default=func.now(),
    )

    updated_at = Column(
        DateTime,
        nullable=False,
        default=func.now(),
        onupdate=func.now(),
    )

    started_at = Column(
        DateTime,
        nullable=True,
    )

    finished_at = Column(
        DateTime,
        nullable=True,
    )