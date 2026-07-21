import pytest

from app.services.document_processing_service import DocumentProcessingService
from app.services.storage_service import StorageService
from app.services.parser_service import ParserService
from app.repositories.document_repository import DocumentRepository
from app.repositories.document_content_repository import DocumentContentRepository
from app.repositories.document_chunk_repository import DocumentChunkRepository
from app.services.chunk_service import ChunkService



@pytest.fixture()
def document_processing_service(tmp_path):
    """
    创建文档处理服务。
    """

    return DocumentProcessingService(
        storage_service=StorageService(str(tmp_path)),
        document_repository=DocumentRepository(),
        document_content_repository=DocumentContentRepository(),
        parser_service=ParserService(),
        chunk_service=ChunkService(),
        document_chunk_repository=DocumentChunkRepository(),
    )


