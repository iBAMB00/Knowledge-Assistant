from sqlalchemy.orm import Session

from app.schemas.document_info import DocumentInfo
from app.services.storage_service import StorageService
from app.repositories.document_repository import DocumentRepository
from app.repositories.document_content_repository import DocumentContentRepository
from app.models.database.document import Document
from app.schemas.document_response import DocumentResponse
from app.constants.document_status import DocumentStatus
from app.services.parser_service import ParserService
from app.models.database.document_content import DocumentContent



class DocumentService:
    """
    文档业务
    负责文档生命周期管理，
    不负责具体文件存储实现。
    """

    def __init__(
            self, 
            storage_service: StorageService,
            document_repository: DocumentRepository,
            document_content_repository: DocumentContentRepository,
            parser_service: ParserService,
        ) -> None:
        """
        初始化文档服务。

        Args:
            storage_service: 文件存储服务。
            document_repository: 文档数据库操作仓库。
        """
        self.storage_service = storage_service
        self.document_repository = document_repository
        self.parser_service = parser_service
        self.document_content_repository = document_content_repository


    def upload_document(
        self,
        db: Session,
        filename: str,
        content: bytes,
    ) -> DocumentInfo:
        """
        保存上传的文档，并返回文档基础信息。

        Args:
            filename: 用户上传时的原始文件名。
            content: 文档的二进制内容。

        Returns:
            上传并完成数据库登记后的文档基础信息。

        Raises:
            ValueError: 文件名为空或文件内容为空时抛出。
        """
        cleaned_filename = filename.strip()

        if not cleaned_filename:
            raise ValueError("filename cannot be empty")

        if not content:
            raise ValueError("file content cannot be empty")

        # 1. 保存文件到存储服务
        stored_result = self.storage_service.save(cleaned_filename, content)
        # 2. 创建数据库对象
        document = Document(
            filename=cleaned_filename,
            stored_name=stored_result.stored_name,
            path=stored_result.path,
            size=len(content),
            status=DocumentStatus.UPLOADED.value,
        )
        # 3. 保存数据库对象
        saved_document = self.document_repository.create(
            db=db, 
            document=document,
        )
        db.commit()
        db.refresh(saved_document)

        # 4. 返回文档基础信息
        return DocumentInfo(
            id=saved_document.id,
            filename=saved_document.filename,
            size=saved_document.size,
            status=saved_document.status,
        )

    def list_documents(
        self,
        db: Session,
    ) -> list[DocumentResponse]:
        """
        查询文档列表。
        """

        documents = self.document_repository.find_all(
            db=db,
        )

        return [
            DocumentResponse(
                id=document.id,
                filename=document.filename,
                stored_name=document.stored_name,
                size=document.size,
                status=document.status,
                created_at=document.created_at,
            )
            for document in documents
        ]

    def delete_document(
        self,
        db: Session,
        document_id: int,
    ) -> None:
        """
        删除指定文档记录。
        """
        document = self.document_repository.find_by_id(
            db=db,
            document_id=document_id,
        )
        if document is None:
            raise ValueError("document not found")
        
        # 1. 删除文件从存储服务
        self.storage_service.delete(document.path)
        # 2. 删除数据库记录
        self.document_repository.delete(
            db=db,
            document=document,
        )
        
        db.commit()



    def process_document(
        self,
        db: Session,
        document_id: int,
    ) -> DocumentResponse:
        """
        同步解析文档并保存解析后的全文。

        流程：
        1. 将状态更新为 parsing 并提交
        2. 从存储服务读取原始文件
        3. 解析文件内容
        4. 保存解析全文并更新为 parsed
        5. 发生异常时回滚并更新为 failed

        Args:
            db:
                数据库会话。

            document_id:
                文档ID。

        Returns:
            处理完成后的文档状态信息。
        """

        document = self.document_repository.find_by_id(
            db=db,
            document_id=document_id,
        )

        if document is None:
            raise ValueError(
                "document not found"
            )


        if document.status not in [
            DocumentStatus.UPLOADED.value,
            DocumentStatus.FAILED.value,
        ]:
            raise ValueError(
                "invalid document status"
            )


        try:
            # 第一阶段：单独提交 parsing，
            # 使其他请求能够看到文档正在处理中。
            self.document_repository.update_status(
                db=db,
                document=document,
                status=DocumentStatus.PARSING.value,
            )
            db.commit()
            db.refresh(document)


            # 解析过程可能耗时，不在数据库事务中持有写操作。
            file_content  = self.storage_service.read(
                document.path,
            )
            
            # 解析服务只接收文件名和二进制内容。
            parse_result = self.parser_service.parse(
                filename=document.filename,
                content=file_content,
            )


            if not parse_result.content.strip():
                raise ValueError(
                    "parsed content is empty"
                )


            # 保存解析后的完整文本和实际解析器类型。
            document_content = DocumentContent(
                document_id=document.id,
                content=parse_result.content,
                parser_type=parse_result.parser_type,
            )

            # 第二阶段：保存解析内容和更新 parsed, 文档可能存在历史解析结果，因此使用新增或覆盖语义。
            # 必须在同一次事务中完成。
            self.document_content_repository.save_or_update(
                db=db,
                document_content=document_content,
            )


            # 5. 文档全文保存成功后标记为已解析。
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
            # 清除当前失败事务，避免 PendingRollbackError。
            db.rollback()
            try:
                document = self.document_repository.find_by_id(
                    db=db,
                    document_id=document_id,
                )

                if document is not None:
                    self.document_repository.update_status(
                        db=db,
                        document=document,
                        status=DocumentStatus.FAILED.value,
                    )

                    db.commit()

            except Exception:
                # FAILED 状态更新本身失败时再次清理会话，
                # 但保留最初的业务异常继续向上传递。
                db.rollback()

            raise
    
    def get_document_content(
        self,
        db: Session,
        document_id: int,
    ) -> str:
        """
        获取指定文档的解析全文。

        Args:
            db:
                数据库会话。

            document_id:
                文档ID。

        Returns:
            文档解析全文。
        """
        document_content = self.document_content_repository.find_by_document_id(
            db=db,
            document_id=document_id,
        )

        if document_content is None:
            raise ValueError(
                "document content not found"
            )

        return document_content.content