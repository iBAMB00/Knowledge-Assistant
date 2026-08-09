from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.dependencies.auth import get_current_user
from app.core.config import get_settings
from app.core.database import get_db
from app.models.database.user import User
from app.repositories.document_repository import DocumentRepository
from app.repositories.knowledge_base_repository import KnowledgeBaseRepository
from app.repositories.document_chunk_repository import DocumentChunkRepository
from app.schemas.retrieval_debug_request import RetrievalDebugRequest
from app.schemas.vector_search_result import VectorSearchResult
from app.services.embedding.factory import EmbeddingFactory
from app.services.bm25_retrieval_service import BM25RetrievalService
from app.services.retrieval_service import RetrievalService
from app.services.knowledge_base_access_policy import (
    KnowledgeBaseAccessPolicy,
    ResourceAccessNotFoundError,
)
from app.services.reranker.factory import RerankerFactory
from app.services.rrf_fusion_service import RRFFusionService
from app.services.vector_store.factory import get_vector_store_components

router = APIRouter(
    prefix="/knowledge/retrieval",
    tags=["Knowledge Retrieval"],
)

settings = get_settings()

embedding_provider = EmbeddingFactory.create()

vector_store = get_vector_store_components().vector_store

document_chunk_repository = DocumentChunkRepository()
knowledge_base_access_policy = KnowledgeBaseAccessPolicy(
    knowledge_base_repository=KnowledgeBaseRepository(),
    document_repository=DocumentRepository(),
)

reranker = (
    RerankerFactory.create()
    if settings.reranker_enabled
    else None
)

retrieval_service = RetrievalService(
    embedding_provider=embedding_provider,
    vector_store=vector_store,
    default_top_k=settings.retrieval_top_k,
    default_candidate_k=settings.retrieval_candidate_k,
    default_score_threshold=settings.retrieval_score_threshold,
    default_per_document_limit=settings.retrieval_per_document_limit,
    document_chunk_repository=document_chunk_repository,
    parent_child_enabled=settings.parent_child_enabled,
    bm25_retriever=BM25RetrievalService(
        document_chunk_repository=document_chunk_repository,
    ),
    rrf_fusion_service=RRFFusionService(
        rank_constant=settings.retrieval_rrf_k,
    ),
    hybrid_enabled=settings.retrieval_hybrid_enabled,
    reranker=reranker,
    reranker_enabled=settings.reranker_enabled,
    reranker_fail_open=settings.reranker_fail_open,
)


@router.post(
    "/debug",
    response_model=list[VectorSearchResult],
)
def debug_retrieval(
    request: RetrievalDebugRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[VectorSearchResult]:
    """
    单文档检索调试接口。
    调试知识库检索结果。

    只执行查询向量化和向量召回，
    不调用大模型生成答案。
    """

    try:
        knowledge_base_access_policy.get_accessible_knowledge_base(
            db=db,
            knowledge_base_id=request.knowledge_base_id,
            user=current_user,
        )
        if request.document_id is not None:
            knowledge_base_access_policy.ensure_document_in_knowledge_base(
                db=db,
                document_id=request.document_id,
                knowledge_base_id=request.knowledge_base_id,
                user=current_user,
            )

        return retrieval_service.retrieve(
            db=db,
            query=request.query,
            top_k=request.top_k,
            candidate_k=request.candidate_k,
            score_threshold=request.score_threshold,
            per_document_limit=request.per_document_limit,
            document_id=request.document_id,
            knowledge_base_id=request.knowledge_base_id,
        )

    except ResourceAccessNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="知识库检索失败",
        ) from exc