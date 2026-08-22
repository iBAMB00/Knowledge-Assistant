from collections.abc import Iterator
import json
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.api.dependencies.auth import get_current_user
from app.api.dependencies.conversation import get_conversation_service
from app.constants.conversation_message_role import ConversationMessageRole
from app.constants.conversation_mode import ConversationMode
from app.core.config import get_settings
from app.core.database import get_db
from app.models.database.user import User
from app.repositories.document_repository import DocumentRepository
from app.repositories.knowledge_base_repository import KnowledgeBaseRepository
from app.repositories.document_chunk_repository import DocumentChunkRepository
from app.schemas.knowledge_chat_request import KnowledgeChatRequest
from app.schemas.knowledge_chat_response import KnowledgeChatResponse
from app.services.conversation_service import (
    ConversationNotFoundError,
    ConversationScopeConflictError,
    ConversationService,
)
from app.services.embedding.factory import EmbeddingFactory
from app.services.bm25_retrieval_service import BM25RetrievalService
from app.services.knowledge_base_access_policy import (
    KnowledgeBaseAccessPolicy,
    ResourceAccessNotFoundError,
)
from app.services.knowledge_chat_service import (
    KnowledgeChatPreparation,
    KnowledgeChatService,
)
from app.services.llm_service import LLMService
from app.services.rag.context_builder import ContextBuilder
from app.services.retrieval_service import RetrievalService
from app.services.reranker.factory import RerankerFactory
from app.services.rrf_fusion_service import RRFFusionService
from app.services.vector_store.factory import get_vector_store_components

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/knowledge",
    tags=["Knowledge Chat"],
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
    default_score_threshold=settings.knowledge_chat_score_threshold,
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

context_builder = ContextBuilder()

llm_service = LLMService()

knowledge_chat_service = KnowledgeChatService(
    retrieval_service=retrieval_service,
    context_builder=context_builder,
    llm_service=llm_service,
    document_chunk_repository=document_chunk_repository,
)


@router.post(
    "/chat",
    response_model=KnowledgeChatResponse,
)
def knowledge_chat(
    request: KnowledgeChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    conversation_service: ConversationService = Depends(
        get_conversation_service
    ),
) -> KnowledgeChatResponse:
    """
    根据知识库内容回答用户问题。
    """

    try:
        knowledge_base_access_policy.get_accessible_knowledge_base(
            db=db, knowledge_base_id=request.knowledge_base_id, user=current_user
        )
        if request.document_id is not None:
            knowledge_base_access_policy.ensure_document_in_knowledge_base(
                db=db,
                document_id=request.document_id,
                knowledge_base_id=request.knowledge_base_id,
                user=current_user,
            )

        question = request.question.strip()
        if not question:
            raise ValueError("question cannot be empty")

        conversation_id = _prepare_conversation_persistence(
            db=db,
            conversation_service=conversation_service,
            current_user=current_user,
            conversation_id=request.conversation_id,
            knowledge_base_id=request.knowledge_base_id,
        )
        _append_conversation_message(
            db=db,
            conversation_service=conversation_service,
            user_id=current_user.id,
            conversation_id=conversation_id,
            role=ConversationMessageRole.USER,
            content=question,
        )

        response = knowledge_chat_service.chat(
            db=db,
            question=question,
            top_k=request.top_k,
            document_id=request.document_id,
            knowledge_base_id=request.knowledge_base_id,
        )

        _append_conversation_message(
            db=db,
            conversation_service=conversation_service,
            user_id=current_user.id,
            conversation_id=conversation_id,
            role=ConversationMessageRole.ASSISTANT,
            content=response.answer,
        )
        return response

    except (ResourceAccessNotFoundError, ConversationNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    except ConversationScopeConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

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
    current_user: User = Depends(get_current_user),
    conversation_service: ConversationService = Depends(
        get_conversation_service
    ),
) -> StreamingResponse:
    """
    根据知识库内容流式回答用户问题。
    """

    try:
        knowledge_base_access_policy.get_accessible_knowledge_base(
            db=db, knowledge_base_id=request.knowledge_base_id, user=current_user
        )
        if request.document_id is not None:
            knowledge_base_access_policy.ensure_document_in_knowledge_base(
                db=db,
                document_id=request.document_id,
                knowledge_base_id=request.knowledge_base_id,
                user=current_user,
            )

        question = request.question.strip()
        if not question:
            raise ValueError("question cannot be empty")

        conversation_id = _prepare_conversation_persistence(
            db=db,
            conversation_service=conversation_service,
            current_user=current_user,
            conversation_id=request.conversation_id,
            knowledge_base_id=request.knowledge_base_id,
        )
        _append_conversation_message(
            db=db,
            conversation_service=conversation_service,
            user_id=current_user.id,
            conversation_id=conversation_id,
            role=ConversationMessageRole.USER,
            content=question,
        )

        preparation = knowledge_chat_service.prepare(
            db=db,
            question=question,
            top_k=request.top_k,
            document_id=request.document_id,
            knowledge_base_id=request.knowledge_base_id,
        )

    except (ResourceAccessNotFoundError, ConversationNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    except ConversationScopeConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

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
            db=db,
            conversation_service=conversation_service,
            user_id=current_user.id,
            conversation_id=conversation_id,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def generate_knowledge_chat_sse(
    preparation: KnowledgeChatPreparation,
    *,
    db: Session | None = None,
    conversation_service: ConversationService | None = None,
    user_id: int | None = None,
    conversation_id: int | None = None,
) -> Iterator[str]:
    """
    生成知识库问答 SSE 事件。

    流正常结束时发送 done；
    流中异常时发送 error；
    调用方关闭生成器时向下关闭业务流。
    """

    content_stream: Iterator[str] | None = None
    answer_parts: list[str] = []

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

            answer_parts.append(content)
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

        answer = "".join(answer_parts).strip()
        if answer:
            _append_conversation_message(
                db=db,
                conversation_service=conversation_service,
                user_id=user_id,
                conversation_id=conversation_id,
                role=ConversationMessageRole.ASSISTANT,
                content=answer,
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


def _prepare_conversation_persistence(
    *,
    db: Session,
    conversation_service: ConversationService,
    current_user: User,
    conversation_id: int | None,
    knowledge_base_id: int,
) -> int | None:
    """可选绑定 RAG Conversation；缺省时保持旧版 stateless API 兼容。"""

    if conversation_id is None:
        return None

    conversation_service.ensure_chat_scope(
        db=db,
        user_id=current_user.id,
        conversation_id=conversation_id,
        mode=ConversationMode.RAG,
        knowledge_base_id=knowledge_base_id,
    )
    return conversation_id


def _append_conversation_message(
    *,
    db: Session | None,
    conversation_service: ConversationService | None,
    user_id: int | None,
    conversation_id: int | None,
    role: ConversationMessageRole,
    content: str,
) -> None:
    """只有显式绑定 Conversation 时才写入用户可见历史。"""

    if (
        db is None
        or conversation_service is None
        or user_id is None
        or conversation_id is None
    ):
        return

    conversation_service.append_message(
        db=db,
        user_id=user_id,
        conversation_id=conversation_id,
        role=role,
        content=content,
    )


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