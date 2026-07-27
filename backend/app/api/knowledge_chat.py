from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.repositories.chunk_embedding_repository import (
    ChunkEmbeddingRepository,
)
from app.schemas.knowledge_chat_request import (
    KnowledgeChatRequest,
)
from app.schemas.knowledge_chat_response import (
    KnowledgeChatResponse,
)
from app.services.embedding.factory import EmbeddingFactory
from app.services.knowledge_chat_service import (
    KnowledgeChatService,
)
from app.services.llm_service import LLMService
from app.services.rag.context_builder import ContextBuilder
from app.services.retrieval_service import RetrievalService
from app.services.vector_store.database import (
    DatabaseVectorStore,
)


router = APIRouter(
    prefix="/knowledge",
    tags=["Knowledge Chat"],
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

context_builder = ContextBuilder()

llm_service = LLMService()

knowledge_chat_service = KnowledgeChatService(
    retrieval_service=retrieval_service,
    context_builder=context_builder,
    llm_service=llm_service,
)


@router.post(
    "/chat",
    response_model=KnowledgeChatResponse,
)
def knowledge_chat(
    request: KnowledgeChatRequest,
    db: Session = Depends(get_db),
) -> KnowledgeChatResponse:
    """
    根据知识库内容回答用户问题。
    """

    try:
        return knowledge_chat_service.chat(
            db=db,
            question=request.question,
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
            detail="知识库问答失败",
        ) from exc