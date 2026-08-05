from collections.abc import Iterator
import json
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.database import get_db
from app.schemas.knowledge_chat_request import KnowledgeChatRequest
from app.schemas.knowledge_chat_response import KnowledgeChatResponse
from app.services.embedding.factory import EmbeddingFactory
from app.services.knowledge_chat_service import (
    KnowledgeChatPreparation,
    KnowledgeChatService,
)
from app.services.llm_service import LLMService
from app.services.rag.context_builder import ContextBuilder
from app.services.retrieval_service import RetrievalService
from app.services.vector_store.factory import get_vector_store_components

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/knowledge",
    tags=["Knowledge Chat"],
)

settings = get_settings()

embedding_provider = EmbeddingFactory.create()

vector_store = get_vector_store_components().vector_store

retrieval_service = RetrievalService(
    embedding_provider=embedding_provider,
    vector_store=vector_store,
    default_top_k=settings.retrieval_top_k,
    default_candidate_k=settings.retrieval_candidate_k,
    default_score_threshold=settings.knowledge_chat_score_threshold,
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
            document_id=request.document_id,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.error(
            "Knowledge chat failed: error_type=%s",
            type(exc).__name__,
        )

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
            document_id=request.document_id,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.error(
            "Knowledge chat preparation failed: "
            "error_type=%s",
            type(exc).__name__,
        )

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
    生成知识库问答 SSE 事件。

    流正常结束时发送 done；
    流中异常时发送 error；
    调用方关闭生成器时向下关闭业务流。
    """

    content_stream: Iterator[str] | None = None

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

        content_stream = (
            knowledge_chat_service.stream_chat(
                preparation
            )
        )

        for content in content_stream:
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

    except GeneratorExit:
        logger.info(
            "Knowledge chat SSE cancelled"
        )
        raise

    except Exception as exc:
        logger.error(
            "Knowledge chat SSE failed: "
            "error_type=%s",
            type(exc).__name__,
        )

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

    finally:
        _close_iterator(content_stream)


def _close_iterator(
    iterator: Any | None,
) -> None:
    """
    尝试关闭流式迭代器。

    关闭失败只记录异常类型，
    不覆盖原始模型或业务异常。
    """

    if iterator is None:
        return

    close_method = getattr(
        iterator,
        "close",
        None,
    )

    if not callable(close_method):
        return

    try:
        close_method()

    except Exception as exc:
        logger.warning(
            "Failed to close knowledge chat stream: "
            "error_type=%s",
            type(exc).__name__,
        )