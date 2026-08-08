from collections.abc import Callable

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.constants.document_status import DocumentStatus
from app.models.database.document import Document
from app.models.database.document_chunk import DocumentChunk
from app.models.database.document_content import DocumentContent
from app.repositories.document_chunk_repository import (
    DocumentChunkRepository,
)
from app.repositories.document_content_repository import (
    DocumentContentRepository,
)
from app.repositories.document_repository import DocumentRepository
from app.schemas.chunk import ChunkResult
from app.schemas.document_response import DocumentResponse
from app.services.chunk_service import ChunkService
from app.services.parser_service import ParserService
from app.services.status_machine import StatusMachine
from app.services.storage_service import StorageService


class DocumentProcessingService:
    """
    文档解析与切片编排服务。

    负责：
    - 读取原始文件
    - 解析并保存文档全文
    - 生成并保存文档切片
    - 管理文档处理状态
    - 管理事务边界

    不负责：
    - 文件上传
    - Embedding生成
    - 向量数据保存
    """

    def __init__(
        self,
        storage_service: StorageService,
        document_repository: DocumentRepository,
        document_content_repository: DocumentContentRepository,
        parser_service: ParserService,
        chunk_service: ChunkService,
        document_chunk_repository: DocumentChunkRepository,
    ) -> None:
        """初始化文档处理服务。"""

        settings = get_settings()
        
        self.storage_service = storage_service
        self.document_repository = document_repository
        self.document_content_repository = (
            document_content_repository
        )
        self.parser_service = parser_service
        self.chunk_service = chunk_service
        self.document_chunk_repository = (
            document_chunk_repository
        )

        self.chunk_strategy = settings.chunk_strategy
        self.parent_child_enabled = settings.parent_child_enabled
        self.parent_child_child_size = settings.parent_child_child_size
        self.parent_child_child_overlap = settings.parent_child_child_overlap

    def process_document(
        self,
        db: Session,
        document_id: int,
        status_callback: Callable[[DocumentStatus], None] | None = None,
    ) -> DocumentResponse:
        """
        解析并切分指定文档。

        支持以下处理路径：

        uploaded / parse_failed
            -> parsing
            -> parsed
            -> chunking
            -> chunked

        parsed / chunk_failed
            -> chunking
            -> chunked
        """

        document = self.document_repository.find_by_id(
            db=db,
            document_id=document_id,
        )

        if document is None:
            raise ValueError("document not found")

        current_status = DocumentStatus(document.status)

        # 已处理状态，直接返回
        if current_status in {
            DocumentStatus.PARSING,
            DocumentStatus.CHUNKING,
        }:
            raise ValueError(
                "document is already being processed"
            )

        if current_status in {
            DocumentStatus.CHUNKED,
            DocumentStatus.EMBEDDING,
            DocumentStatus.EMBEDDING_FAILED,
            DocumentStatus.COMPLETED,
        }:
            return self._build_document_response(document)

        if current_status in {
            DocumentStatus.UPLOADED,
            DocumentStatus.PARSE_FAILED,
        }:
            document_content = self._parse_document(
                db=db,
                document=document,
                status_callback=status_callback,
            )

        elif current_status in {
            DocumentStatus.PARSED,
            DocumentStatus.CHUNK_FAILED,
        }:
            document_content = (
                self.document_content_repository
                .find_by_document_id(
                    db=db,
                    document_id=document.id,
                )
            )

            if document_content is None:
                raise ValueError(
                    "document content not found"
                )

        else:
            raise ValueError(
                "invalid document status: "
                f"{current_status.value}"
            )

        self._chunk_document(
            db=db,
            document=document,
            document_content=document_content,
            status_callback=status_callback,
        )

        db.refresh(document)

        return self._build_document_response(document)

    def _parse_document(
        self,
        db: Session,
        document: Document,
        status_callback: Callable[[DocumentStatus], None] | None,
    ) -> DocumentContent:
        """
        执行文档解析阶段。

        parsing状态单独提交；
        解析内容和parsed状态在同一事务中提交。
        """

        try:
            StatusMachine.transition_document(
                document=document,
                target_status=DocumentStatus.PARSING,
            )

            db.commit()
            db.refresh(document)
            self._notify_status(
                status_callback=status_callback,
                status=DocumentStatus.PARSING,
            )

            file_content = self.storage_service.read(
                document.path
            )

            parse_result = self.parser_service.parse(
                filename=document.filename,
                content=file_content,
            )

            if not parse_result.content.strip():
                raise ValueError(
                    "parsed content is empty"
                )

            document_content = DocumentContent(
                document_id=document.id,
                content=parse_result.content,
                parser_type=parse_result.parser_type,
                parser_version=parse_result.parser_version,
            )

            saved_content = (
                self.document_content_repository
                .save_or_update(
                    db=db,
                    document_content=document_content,
                )
            )

            StatusMachine.transition_document(
                document=document,
                target_status=DocumentStatus.PARSED,
            )

            db.commit()
            db.refresh(saved_content)

            return saved_content

        except Exception:
            self._safe_mark_document_failed(
                db=db,
                document_id=document.id,
                failed_status=DocumentStatus.PARSE_FAILED,
            )
            raise

    def _chunk_document(
        self,
        db: Session,
        document: Document,
        document_content: DocumentContent,
        status_callback: Callable[[DocumentStatus], None] | None,
    ) -> None:
        """
        执行文档切片阶段。

        chunking状态单独提交；
        Chunk保存和chunked状态在同一事务中提交。
        """

        try:
            StatusMachine.transition_document(
                document=document,
                target_status=DocumentStatus.CHUNKING,
            )

            db.commit()
            db.refresh(document)
            self._notify_status(
                status_callback=status_callback,
                status=DocumentStatus.CHUNKING,
            )

            parent_chunks = self.chunk_service.split(
                content=document_content.content,
                strategy_name=self.chunk_strategy,
                metadata={
                    "document_id": document.id,
                    "document_content_id": (
                        document_content.id
                    ),
                    "chunk_strategy": self.chunk_strategy,
                    "chunk_role": "parent",
                },
            )

            if not parent_chunks:
                raise ValueError(
                    "document chunks are empty"
                )

            # 切片重试时先删除原有切片，
            # 避免chunk_index唯一约束冲突。
            self.document_chunk_repository\
                .delete_by_document_content_id(
                    db=db,
                    document_content_id=document_content.id,
                )

            document_chunks = self._build_document_chunks(
                document_content_id=document_content.id,
                chunks=parent_chunks,
            )

            saved_parent_chunks = (
                self.document_chunk_repository.save_all(
                    db=db,
                    chunks=document_chunks,
                )
            )

            if self.parent_child_enabled:
                child_chunks = self._build_child_document_chunks(
                    document_content_id=document_content.id,
                    parent_chunks=saved_parent_chunks,
                )

                if child_chunks:
                    self.document_chunk_repository.save_all(
                        db=db,
                        chunks=child_chunks,
                    )

            StatusMachine.transition_document(
                document=document,
                target_status=DocumentStatus.CHUNKED,
            )

            db.commit()

        except Exception:
            self._safe_mark_document_failed(
                db=db,
                document_id=document.id,
                failed_status=DocumentStatus.CHUNK_FAILED,
            )
            raise

    def _safe_mark_document_failed(
        self,
        db: Session,
        document_id: int,
        failed_status: DocumentStatus,
    ) -> None:
        """
        回滚当前事务并安全记录失败状态。

        记录失败状态本身出错时，
        不覆盖原始业务异常。
        """

        db.rollback()
        db.expire_all()

        try:
            document = self.document_repository.find_by_id(
                db=db,
                document_id=document_id,
            )

            if document is None:
                return

            StatusMachine.transition_document(
                document=document,
                target_status=failed_status,
            )

            db.commit()

        except Exception:
            db.rollback()

    @staticmethod
    def _notify_status(
        status_callback: Callable[[DocumentStatus], None] | None,
        status: DocumentStatus,
    ) -> None:
        """
        将已提交的文档业务状态通知给上层编排器。
        """

        if status_callback is not None:
            status_callback(status)

    def _build_document_chunks(
        self,
        document_content_id: int,
        chunks: list[ChunkResult],
    ) -> list[DocumentChunk]:
        """
        将ChunkResult转换为DocumentChunk。
        """

        return [
            DocumentChunk(
                document_content_id=document_content_id,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                token_count=chunk.token_count,
                chunk_strategy=chunk.metadata.get(
                    "chunk_strategy",
                    "recursive_character",
                ),
                chunk_metadata=chunk.metadata,
            )
            for chunk in chunks
        ]


    def _build_child_document_chunks(
        self,
        document_content_id: int,
        parent_chunks: list[DocumentChunk],
    ) -> list[DocumentChunk]:
        """在每个Parent内部生成更细粒度的Child Chunk。"""

        child_document_chunks: list[DocumentChunk] = []
        next_chunk_index = len(parent_chunks)

        for parent_chunk in parent_chunks:
            child_results = self.chunk_service.split(
                content=parent_chunk.content,
                strategy_name=self.chunk_strategy,
                chunk_size=self.parent_child_child_size,
                chunk_overlap=self.parent_child_child_overlap,
                metadata={
                    "document_content_id": document_content_id,
                    "chunk_strategy": self.chunk_strategy,
                    "chunk_role": "child",
                    "parent_chunk_id": parent_chunk.id,
                    "parent_chunk_index": parent_chunk.chunk_index,
                },
            )

            for child_result in child_results:
                child_document_chunks.append(
                    DocumentChunk(
                        document_content_id=document_content_id,
                        chunk_index=next_chunk_index,
                        content=child_result.content,
                        token_count=child_result.token_count,
                        chunk_strategy=self.chunk_strategy,
                        chunk_metadata=child_result.metadata,
                        parent_chunk_id=parent_chunk.id,
                    )
                )
                next_chunk_index += 1

        return child_document_chunks

    def _build_document_response(
        self,
        document: Document,
    ) -> DocumentResponse:
        """
        将Document转换为响应对象。
        """

        return DocumentResponse(
            id=document.id,
            filename=document.filename,
            stored_name=document.stored_name,
            size=document.size,
            status=document.status,
            created_at=document.created_at,
        )