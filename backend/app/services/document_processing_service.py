from sqlalchemy.orm import Session

from app.constants.document_status import DocumentStatus
from app.models.database.document_content import DocumentContent
from app.repositories.document_content_repository import DocumentContentRepository
from app.repositories.document_repository import DocumentRepository
from app.schemas.document_response import DocumentResponse
from app.services.parser_service import ParserService
from app.services.storage_service import StorageService
from app.repositories.document_chunk_repository import DocumentChunkRepository
from app.models.database.document_chunk import DocumentChunk
from app.services.chunk_service import ChunkService
from app.schemas.chunk import ChunkResult



class DocumentProcessingService:
    """
    文档处理编排服务。

    负责文档解析、切片、向量化等知识加工流程。
    当前阶段仅实现文档解析和全文持久化。
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
        """
        初始化文档处理服务。

        Args:
            storage_service:
                文件存储服务。

            document_repository:
                文档元数据仓库。

            document_content_repository:
                文档解析全文仓库。

            parser_service:
                文档解析服务。
        """

        self.storage_service = storage_service
        self.document_repository = document_repository
        self.document_content_repository = (
            document_content_repository
        )
        self.parser_service = parser_service

    def process_document(
        self,
        db: Session,
        document_id: int,
    ) -> DocumentResponse:
        """
        同步解析文档并保存解析全文。

        当前处理流程：
        uploaded/failed
            -> parsing
            -> parsed

        Args:
            db:
                数据库会话。

            document_id:
                待处理文档ID。

        Returns:
            处理完成后的文档状态信息。

        Raises:
            ValueError:
                文档不存在、状态不允许处理，
                或解析结果为空。
        """

        document = self.document_repository.find_by_id(
            db=db,
            document_id=document_id,
        )

        if document is None:
            raise ValueError("document not found")

        if document.status not in [
            DocumentStatus.UPLOADED.value,
            DocumentStatus.FAILED.value,
        ]:
            raise ValueError(
                "invalid document status"
            )

        try:
            # 第一阶段：提交 parsing 状态，
            # 让其他请求能够观察到文档正在处理。
            self.document_repository.update_status(
                db=db,
                document=document,
                status=DocumentStatus.PARSING.value,
            )

            db.commit()
            db.refresh(document)

            # 文件读取由存储层负责，
            # 处理服务不直接访问本地文件系统。
            file_content = self.storage_service.read(
                document.path,
            )

            # 解析层只接收文件名和二进制内容。
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
            )

            # 解析全文写入和 parsed 状态更新
            # 必须在同一个数据库事务中完成。
            self.document_content_repository.save_or_update(
                db=db,
                document_content=document_content,
            )

            # 第二阶段：切分解析全文。
            # 切分策略为 recursive_character。  （暂时写死，因为只有一个策略）
            # 元他元数据为 document_id、document_content_id、chunk_strategy。
            chunks = self.chunk_service.split(
                content=document_content.content,
                strategy_name="recursive_character",
                metadata={
                    "document_id": document.id,
                    "document_content_id": document_content.id,                                                             
                    "chunk_strategy": "recursive_character",
                },
            )


            document_chunks = self._build_document_chunks(
                document_content_id=document_content.id,
                chunks=chunks,
            )
            self.document_chunk_repository.save_all(
                db=db,
                chunks=document_chunks,
            )

            self.document_repository.update_status(
                db=db,
                document=document,
                status=DocumentStatus.PARSED.value,
            )

            db.commit()
            db.refresh(document)

            return DocumentResponse(
                id=document.id,
                filename=document.filename,
                stored_name=document.stored_name,
                size=document.size,
                status=document.status,
                created_at=document.created_at,
            )

        except Exception:
            # 先清理失败事务，否则可能触发
            # SQLAlchemy PendingRollbackError。
            db.rollback()

            try:
                failed_document = (
                    self.document_repository.find_by_id(
                        db=db,
                        document_id=document_id,
                    )
                )

                if failed_document is not None:
                    self.document_repository.update_status(
                        db=db,
                        document=failed_document,
                        status=DocumentStatus.FAILED.value,
                    )

                    db.commit()

            except Exception:
                # 更新失败状态本身失败时清理会话，
                # 保留最初的业务异常。
                db.rollback()

            raise
    
    def _build_document_chunks(
        self,
        document_content_id: int,
        chunks: list[ChunkResult],
    ) -> list[DocumentChunk]:
        """
        将 ChunkResult 转换为数据库切片模型。

        ChunkService负责生成切片结果，
        本方法负责业务对象到数据库对象转换。
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