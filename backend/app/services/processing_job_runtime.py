from functools import lru_cache

from app.core.database import SessionLocal
from app.repositories.chunk_embedding_repository import ChunkEmbeddingRepository
from app.repositories.document_chunk_repository import DocumentChunkRepository
from app.repositories.document_content_repository import DocumentContentRepository
from app.repositories.document_repository import DocumentRepository
from app.repositories.processing_job_repository import ProcessingJobRepository
from app.services.chunk_service import ChunkService
from app.services.document_processing_service import DocumentProcessingService
from app.services.embedding.factory import EmbeddingFactory
from app.services.embedding_service import EmbeddingService
from app.services.parser_service import ParserService
from app.services.processing_job_executor import ProcessingJobExecutor
from app.services.processing_job_runner import ProcessingJobRunner
from app.services.processing_job_service import ProcessingJobService
from app.services.storage_service import StorageService
from app.services.vector_index_service import VectorIndexService
from app.services.vector_store.factory import get_vector_store_components


@lru_cache
def get_processing_job_executor() -> ProcessingJobExecutor:
    """构建当前进程共享的文档处理任务执行器。"""
    document_repository = DocumentRepository()
    document_chunk_repository = DocumentChunkRepository()
    chunk_embedding_repository = ChunkEmbeddingRepository()

    processing_job_service = ProcessingJobService(
        document_repository=document_repository,
        processing_job_repository=ProcessingJobRepository(),
    )

    document_processing_service = DocumentProcessingService(
        storage_service=StorageService(),
        document_repository=document_repository,
        document_content_repository=DocumentContentRepository(),
        parser_service=ParserService(),
        chunk_service=ChunkService(),
        document_chunk_repository=document_chunk_repository,
    )

    embedding_service = EmbeddingService(
        document_repository=document_repository,
        document_chunk_repository=document_chunk_repository,
        chunk_embedding_repository=chunk_embedding_repository,
        embedding_provider=EmbeddingFactory.create(),
    )

    vector_index_service = None
    vector_index = get_vector_store_components().vector_index

    if vector_index is not None:
        vector_index_service = VectorIndexService(
            document_repository=document_repository,
            chunk_embedding_repository=chunk_embedding_repository,
            vector_index=vector_index,
        )

    return ProcessingJobExecutor(
        document_repository=document_repository,
        processing_job_service=processing_job_service,
        document_processing_service=document_processing_service,
        embedding_service=embedding_service,
        vector_index_service=vector_index_service,
    )


@lru_cache
def get_processing_job_runner() -> ProcessingJobRunner:
    """构建由 Worker 使用的独立 Session 任务 Runner。"""
    return ProcessingJobRunner(
        session_factory=SessionLocal,
        executor=get_processing_job_executor(),
    )
