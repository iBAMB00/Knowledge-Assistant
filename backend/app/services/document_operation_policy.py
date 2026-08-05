from sqlalchemy.orm import Session

from app.repositories.processing_job_repository import (
    ProcessingJobRepository,
)


class DocumentOperationConflictError(ValueError):
    """
    文档当前状态与活动任务不允许执行指定操作。
    """


class DocumentOperationPolicy:
    """
    文档操作策略。

    负责集中表达文档操作与处理任务之间的冲突规则，
    不负责HTTP异常转换、数据库提交或具体删除动作。
    """

    def __init__(
        self,
        processing_job_repository: ProcessingJobRepository,
    ) -> None:
        self.processing_job_repository = processing_job_repository

    def ensure_can_delete(
        self,
        db: Session,
        document_id: int,
    ) -> None:
        """
        校验文档当前是否允许删除。

        pending或running任务都属于活动任务；
        存在活动任务时必须拒绝删除，避免文件、Chunk、
        Embedding和外部向量索引在处理过程中被清理。
        """

        if (
            self.processing_job_repository
            .exists_active_by_document_id(
                db=db,
                document_id=document_id,
            )
        ):
            raise DocumentOperationConflictError(
                "document has an active processing job "
                "and cannot be deleted"
            )
