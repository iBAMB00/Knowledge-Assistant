from sqlalchemy.orm import Session

from app.constants.document_status import DocumentStatus
from app.constants.embedding_status import EmbeddingStatus
from app.models.database.processing_job import ProcessingJob
from app.repositories.document_chunk_repository import DocumentChunkRepository
from app.repositories.document_repository import DocumentRepository
from app.services.status_machine import StatusMachine


class ProcessingJobRecoveryService:
    """把 Worker 异常中断留下的处理中状态恢复为可重试状态。"""

    def __init__(
        self,
        document_repository: DocumentRepository,
        document_chunk_repository: DocumentChunkRepository,
    ) -> None:
        self.document_repository = document_repository
        self.document_chunk_repository = document_chunk_repository

    def prepare_for_resume(self, db: Session, job: ProcessingJob) -> None:
        """恢复当前任务对应文档的中间状态，并保留已经成功的数据。"""
        document = self.document_repository.find_by_id(
            db=db,
            document_id=job.document_id,
        )
        if document is None:
            raise ValueError("document not found")

        current_status = DocumentStatus(document.status)

        if current_status == DocumentStatus.PARSING:
            StatusMachine.transition_document(document, DocumentStatus.PARSE_FAILED)
        elif current_status == DocumentStatus.CHUNKING:
            StatusMachine.transition_document(document, DocumentStatus.CHUNK_FAILED)
        elif current_status == DocumentStatus.EMBEDDING:
            chunks = self.document_chunk_repository.find_by_document_id(
                db=db,
                document_id=document.id,
            )
            for chunk in chunks:
                if EmbeddingStatus(chunk.embedding_status) == EmbeddingStatus.PROCESSING:
                    StatusMachine.transition_embedding(chunk, EmbeddingStatus.FAILED)
            StatusMachine.transition_document(document, DocumentStatus.EMBEDDING_FAILED)

        db.commit()
