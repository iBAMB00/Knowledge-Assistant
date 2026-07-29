import json
from collections.abc import Iterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.config import get_settings
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
    KnowledgeChatPreparation,
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

settings = get_settings()

embedding_provider = EmbeddingFactory.create()

chunk_embedding_repository = ChunkEmbeddingRepository()

vector_store = DatabaseVectorStore(
    chunk_embedding_repository=chunk_embedding_repository,
)

retrieval_service = RetrievalService(
    embedding_provider=embedding_provider,
    vector_store=vector_store,
    default_top_k=settings.retrieval_top_k,
    default_candidate_k=settings.retrieval_candidate_k,
    default_score_threshold=settings.retrieval_score_threshold,
    default_per_document_limit=settings.retrieval_per_document_limit,
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


@router.post("/chat/stream")
def stream_knowledge_chat(
    request: KnowledgeChatRequest,
    db: Session = Depends(get_db),
) -> StreamingResponse:
    """
    根据知识库内容流式回答用户问题。
    """

    try:
        preparation = knowledge_chat_service.prepare(
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
            detail="知识库检索失败",
        ) from exc

    return StreamingResponse(
        generate_knowledge_chat_sse(
            preparation=preparation,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def generate_knowledge_chat_sse(
    preparation: KnowledgeChatPreparation,
) -> Iterator[str]:
    """
    生成知识库问答SSE事件。
    """

    try:
        metadata = json.dumps(
            {
                "sources": jsonable_encoder(
                    preparation.sources
                )
            },
            ensure_ascii=False,
        )

        yield (
            "event: metadata\n"
            f"data: {metadata}\n\n"
        )

        for content in (
            knowledge_chat_service.stream_chat(
                preparation
            )
        ):
            if not content:
                continue

            data = json.dumps(
                {
                    "content": content,
                },
                ensure_ascii=False,
            )

            yield (
                "event: message\n"
                f"data: {data}\n\n"
            )

        yield "event: done\ndata: {}\n\n"

    except Exception:
        data = json.dumps(
            {
                "message": "知识库问答失败",
            },
            ensure_ascii=False,
        )

        yield (
            "event: error\n"
            f"data: {data}\n\n"
        )