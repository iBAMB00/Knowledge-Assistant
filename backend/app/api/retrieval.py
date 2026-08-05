from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.database import get_db
from app.schemas.retrieval_debug_request import RetrievalDebugRequest
from app.schemas.vector_search_result import VectorSearchResult
from app.services.embedding.factory import EmbeddingFactory
from app.services.retrieval_service import RetrievalService
from app.services.vector_store.factory import get_vector_store_components

router = APIRouter(
    prefix="/knowledge/retrieval",
    tags=["Knowledge Retrieval"],
)

settings = get_settings()

embedding_provider = EmbeddingFactory.create()

vector_store = get_vector_store_components().vector_store

retrieval_service = RetrievalService(
    embedding_provider=embedding_provider,
    vector_store=vector_store,
    default_top_k=settings.retrieval_top_k,
    default_candidate_k=settings.retrieval_candidate_k,
    default_score_threshold=settings.retrieval_score_threshold,
    default_per_document_limit=settings.retrieval_per_document_limit,
)


@router.post(
    "/debug",
    response_model=list[VectorSearchResult],
)
def debug_retrieval(
    request: RetrievalDebugRequest,
    db: Session = Depends(get_db),
) -> list[VectorSearchResult]:
    """
    单文档检索调试接口。
    调试知识库检索结果。

    只执行查询向量化和向量召回，
    不调用大模型生成答案。
    """

    try:
        return retrieval_service.retrieve(
            db=db,
            query=request.query,
            top_k=request.top_k,
            candidate_k=request.candidate_k,
            score_threshold=request.score_threshold,
            per_document_limit=request.per_document_limit,
            document_id=request.document_id,
        )

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