from typing import Final

from app.constants.document_status import DocumentStatus
from app.constants.embedding_status import EmbeddingStatus
from app.constants.processing_job_stage import ProcessingJobStage
from app.constants.processing_job_status import ProcessingJobStatus
from app.models.database.document import Document
from app.models.database.document_chunk import DocumentChunk
from app.models.database.processing_job import ProcessingJob


class InvalidStatusTransitionError(ValueError):
    """
    状态流转不符合业务规则时抛出的异常。
    """


DOCUMENT_STATUS_TRANSITIONS: Final[
    dict[DocumentStatus, frozenset[DocumentStatus]]
] = {
    DocumentStatus.UPLOADED: frozenset({
        DocumentStatus.PARSING,
    }),
    DocumentStatus.PARSING: frozenset({
        DocumentStatus.PARSED,
        DocumentStatus.PARSE_FAILED,
    }),
    DocumentStatus.PARSE_FAILED: frozenset({
        DocumentStatus.PARSING,
    }),
    DocumentStatus.PARSED: frozenset({
        DocumentStatus.CHUNKING,
    }),
    DocumentStatus.CHUNKING: frozenset({
        DocumentStatus.CHUNKED,
        DocumentStatus.CHUNK_FAILED,
    }),
    DocumentStatus.CHUNK_FAILED: frozenset({
        DocumentStatus.CHUNKING,
    }),
    DocumentStatus.CHUNKED: frozenset({
        DocumentStatus.EMBEDDING,
    }),
    DocumentStatus.EMBEDDING: frozenset({
        DocumentStatus.COMPLETED,
        DocumentStatus.EMBEDDING_FAILED,
    }),
    DocumentStatus.EMBEDDING_FAILED: frozenset({
        DocumentStatus.EMBEDDING,
    }),
    DocumentStatus.COMPLETED: frozenset(),
}


EMBEDDING_STATUS_TRANSITIONS: Final[
    dict[EmbeddingStatus, frozenset[EmbeddingStatus]]
] = {
    EmbeddingStatus.PENDING: frozenset({
        EmbeddingStatus.PROCESSING,
    }),
    EmbeddingStatus.PROCESSING: frozenset({
        EmbeddingStatus.COMPLETED,
        EmbeddingStatus.FAILED,
    }),
    EmbeddingStatus.FAILED: frozenset({
        EmbeddingStatus.PROCESSING,
    }),
    EmbeddingStatus.COMPLETED: frozenset(),
}


PROCESSING_JOB_STATUS_TRANSITIONS: Final[
    dict[
        ProcessingJobStatus,
        frozenset[ProcessingJobStatus],
    ]
] = {
    ProcessingJobStatus.PENDING: frozenset({
        ProcessingJobStatus.RUNNING,
        ProcessingJobStatus.FAILED,
    }),
    ProcessingJobStatus.RUNNING: frozenset({
        ProcessingJobStatus.SUCCEEDED,
        ProcessingJobStatus.FAILED,
    }),
    ProcessingJobStatus.SUCCEEDED: frozenset(),
    ProcessingJobStatus.FAILED: frozenset(),
}


PROCESSING_JOB_STAGE_TRANSITIONS: Final[
    dict[
        ProcessingJobStage,
        frozenset[ProcessingJobStage],
    ]
] = {
    ProcessingJobStage.QUEUED: frozenset({
        ProcessingJobStage.PARSING,
        ProcessingJobStage.CHUNKING,
        ProcessingJobStage.EMBEDDING,
        ProcessingJobStage.INDEXING,
    }),
    ProcessingJobStage.PARSING: frozenset({
        ProcessingJobStage.CHUNKING,
    }),
    ProcessingJobStage.CHUNKING: frozenset({
        ProcessingJobStage.EMBEDDING,
        ProcessingJobStage.FINALIZING,
    }),
    ProcessingJobStage.EMBEDDING: frozenset({
        ProcessingJobStage.INDEXING,
        ProcessingJobStage.FINALIZING,
    }),
    ProcessingJobStage.INDEXING: frozenset({
        ProcessingJobStage.FINALIZING,
    }),
    ProcessingJobStage.FINALIZING: frozenset({
        ProcessingJobStage.COMPLETED,
    }),
    ProcessingJobStage.COMPLETED: frozenset(),
}


class StatusMachine:
    """
    文档、Chunk和处理任务状态转换器。

    只负责验证并修改状态，不负责数据库commit。
    事务边界仍由业务Service管理。
    """

    @classmethod
    def transition_document(
        cls,
        document: Document,
        target_status: DocumentStatus,
    ) -> Document:
        """
        转换文档状态。
        """

        current_status = DocumentStatus(document.status)

        if current_status == target_status:
            return document

        allowed_statuses = DOCUMENT_STATUS_TRANSITIONS.get(
            current_status,
            frozenset(),
        )

        if target_status not in allowed_statuses:
            raise InvalidStatusTransitionError(
                "非法文档状态流转："
                f"{current_status.value} -> {target_status.value}"
            )

        document.status = target_status.value
        return document

    @classmethod
    def transition_embedding(
        cls,
        chunk: DocumentChunk,
        target_status: EmbeddingStatus,
    ) -> DocumentChunk:
        """
        转换Chunk向量化状态。
        """

        current_status = EmbeddingStatus(
            chunk.embedding_status
        )

        if current_status == target_status:
            return chunk

        allowed_statuses = EMBEDDING_STATUS_TRANSITIONS.get(
            current_status,
            frozenset(),
        )

        if target_status not in allowed_statuses:
            raise InvalidStatusTransitionError(
                "非法Chunk向量状态流转："
                f"{current_status.value} -> {target_status.value}"
            )

        chunk.embedding_status = target_status.value
        return chunk

    @classmethod
    def transition_processing_job(
        cls,
        job: ProcessingJob,
        target_status: ProcessingJobStatus,
    ) -> ProcessingJob:
        """
        转换文档处理任务状态。

        succeeded和failed是终态；
        重试需要创建新的ProcessingJob。
        """

        current_status = ProcessingJobStatus(job.status)

        if current_status == target_status:
            return job

        allowed_statuses = PROCESSING_JOB_STATUS_TRANSITIONS.get(
            current_status,
            frozenset(),
        )

        if target_status not in allowed_statuses:
            raise InvalidStatusTransitionError(
                "非法任务状态流转："
                f"{current_status.value} -> "
                f"{target_status.value}"
            )

        job.status = target_status.value
        return job

    @classmethod
    def transition_processing_job_stage(
        cls,
        job: ProcessingJob,
        target_stage: ProcessingJobStage,
    ) -> ProcessingJob:
        """
        转换文档处理任务业务阶段。

        阶段只允许向前推进，失败时不单独改变阶段。
        """

        current_stage = ProcessingJobStage(job.stage)

        if current_stage == target_stage:
            return job

        allowed_stages = PROCESSING_JOB_STAGE_TRANSITIONS.get(
            current_stage,
            frozenset(),
        )

        if target_stage not in allowed_stages:
            raise InvalidStatusTransitionError(
                "非法任务阶段流转："
                f"{current_stage.value} -> "
                f"{target_stage.value}"
            )

        job.stage = target_stage.value
        return job
