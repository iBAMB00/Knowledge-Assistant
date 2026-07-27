from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.repositories.chunk_embedding_repository import (
    ChunkEmbeddingRepository,
)
from app.schemas.retrieval_debug_request import (
    RetrievalDebugRequest,
)
from app.schemas.vector_search_result import (
    VectorSearchResult,
)
from app.services.embedding.factory import EmbeddingFactory
from app.services.retrieval_service import RetrievalService
from app.services.vector_store.database import (
    DatabaseVectorStore,
)


router = APIRouter(
    prefix="/knowledge/retrieval",
    tags=["Knowledge Retrieval"],
)


embedding_provider = EmbeddingFactory.create()

chunk_embedding_repository = ChunkEmbeddingRepository()

vector_store = DatabaseVectorStore(
    chunk_embedding_repository=chunk_embedding_repository,
)

retrieval_service = RetrievalService(
    embedding_provider=embedding_provider,
    vector_store=vector_store,
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
    调试知识库检索结果。

    只执行查询向量化和向量召回，
    不调用大模型生成答案。
    """

    try:
        return retrieval_service.retrieve(
            db=db,
            query=request.query,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
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