from collections.abc import Callable
import logging
from typing import Any

from sqlalchemy.orm import Session

from app.constants.document_status import DocumentStatus
from app.core.config import get_settings
from app.models.database.document import Document
from app.models.database.document_chunk import DocumentChunk
from app.models.database.document_content import DocumentContent
from app.repositories.document_chunk_repository import DocumentChunkRepository
from app.repositories.document_content_repository import DocumentContentRepository
from app.repositories.document_repository import DocumentRepository
from app.schemas.chunk import ChunkResult
from app.schemas.document_response import DocumentResponse
from app.services.chunk_service import ChunkService
from app.services.parser_service import ParserService
from app.services.status_machine import StatusMachine
from app.services.storage_service import StorageService


logger = logging.getLogger(__name__)


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
        self.structure_aware_parent_enabled = (
            settings.structure_aware_parent_enabled
        )
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
                structure_metadata=(
                    parse_result.to_structure_metadata()
                ),
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

            parent_chunks = self._build_parent_chunk_results(
                document=document,
                document_content=document_content,
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
                chunk_metadata=self._build_persisted_chunk_metadata(
                    chunk=chunk,
                ),
            )
            for chunk in chunks
        ]


    def _build_parent_chunk_results(
        self,
        document: Document,
        document_content: DocumentContent,
    ) -> list[ChunkResult]:
        """
        生成 Parent Chunk。

        有可靠 Section 结构时优先按章节边界切分；结构缺失、关闭或
        校验失败时降级到原有全文 recursive-character 逻辑。
        """

        base_metadata = {
            "document_id": document.id,
            "document_content_id": document_content.id,
            "chunk_strategy": self.chunk_strategy,
            "chunk_role": "parent",
        }

        if (
            self.structure_aware_parent_enabled
            and document_content.structure_metadata
        ):
            structured_chunks = (
                self.chunk_service.split_parent_by_structure(
                    content=document_content.content,
                    strategy_name=self.chunk_strategy,
                    structure_metadata=(
                        document_content.structure_metadata
                    ),
                    metadata=base_metadata,
                )
            )

            if structured_chunks:
                logger.info(
                    "structure-aware parent chunking applied: "
                    "document_id=%s, parents=%d",
                    document.id,
                    len(structured_chunks),
                )
                return structured_chunks

            logger.warning(
                "structure-aware parent chunking fallback: "
                "document_id=%s, reason=invalid_structure_metadata",
                document.id,
            )

        return self.chunk_service.split(
            content=document_content.content,
            strategy_name=self.chunk_strategy,
            metadata={
                **base_metadata,
                "structure_aware": False,
                "chunk_boundary_mode": "document",
            },
        )


    @staticmethod
    def _build_persisted_chunk_metadata(
        chunk: ChunkResult,
    ) -> dict[str, Any]:
        """将算法 offset 一并保存，便于后续结构追踪与调试。"""

        metadata = dict(chunk.metadata)
        metadata.setdefault(
            "document_start_offset",
            chunk.start_offset,
        )
        metadata.setdefault(
            "document_end_offset",
            chunk.end_offset,
        )
        return metadata


    def _build_child_document_chunks(
        self,
        document_content_id: int,
        parent_chunks: list[DocumentChunk],
    ) -> list[DocumentChunk]:
        """在每个Parent内部生成更细粒度的Child Chunk。"""

        child_document_chunks: list[DocumentChunk] = []
        next_chunk_index = len(parent_chunks)

        for parent_chunk in parent_chunks:
            parent_metadata = parent_chunk.chunk_metadata or {}
            child_metadata = {
                "document_content_id": document_content_id,
                "chunk_strategy": self.chunk_strategy,
                "chunk_role": "child",
                "chunk_boundary_mode": "parent",
                "parent_chunk_id": parent_chunk.id,
                "parent_chunk_index": parent_chunk.chunk_index,
            }
            self._copy_parent_structure_metadata(
                parent_metadata=parent_metadata,
                child_metadata=child_metadata,
            )

            child_results = self.chunk_service.split(
                content=parent_chunk.content,
                strategy_name=self.chunk_strategy,
                chunk_size=self.parent_child_child_size,
                chunk_overlap=self.parent_child_child_overlap,
                metadata=child_metadata,
            )

            parent_document_start = parent_metadata.get(
                "document_start_offset"
            )

            for child_result in child_results:
                persisted_child_metadata = dict(
                    child_result.metadata
                )

                if isinstance(parent_document_start, int):
                    persisted_child_metadata[
                        "document_start_offset"
                    ] = (
                        parent_document_start
                        + child_result.start_offset
                    )
                    persisted_child_metadata[
                        "document_end_offset"
                    ] = (
                        parent_document_start
                        + child_result.end_offset
                    )

                child_document_chunks.append(
                    DocumentChunk(
                        document_content_id=document_content_id,
                        chunk_index=next_chunk_index,
                        content=child_result.content,
                        token_count=child_result.token_count,
                        chunk_strategy=self.chunk_strategy,
                        chunk_metadata=persisted_child_metadata,
                        parent_chunk_id=parent_chunk.id,
                    )
                )
                next_chunk_index += 1

        return child_document_chunks

    @staticmethod
    def _copy_parent_structure_metadata(
        parent_metadata: dict[str, Any],
        child_metadata: dict[str, Any],
    ) -> None:
        """将 Parent 的章节语义复制给 Child，不复制正文。"""

        keys = (
            "structure_aware",
            "source_format",
            "section_index",
            "section_title",
            "section_level",
            "heading_path",
            "section_start_offset",
            "section_end_offset",
            "section_part_index",
            "section_part_count",
        )

        for key in keys:
            if key in parent_metadata:
                child_metadata[key] = parent_metadata[key]


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