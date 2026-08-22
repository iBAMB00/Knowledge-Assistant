from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from sqlalchemy.orm import Session

from app.api.dependencies.auth import get_current_user
from app.constants.conversation_mode import ConversationMode
from app.core.database import get_db
from app.models.database.user import User
from app.repositories.conversation_message_repository import (
    ConversationMessageRepository,
)
from app.repositories.conversation_repository import ConversationRepository
from app.repositories.document_repository import DocumentRepository
from app.repositories.knowledge_base_repository import KnowledgeBaseRepository
from app.schemas.conversation_create_request import ConversationCreateRequest
from app.schemas.conversation_response import (
    ConversationMessageResponse,
    ConversationResponse,
)
from app.services.conversation_service import (
    ConversationNotFoundError,
    ConversationService,
)
from app.services.knowledge_base_access_policy import (
    KnowledgeBaseAccessPolicy,
    ResourceAccessNotFoundError,
)


router = APIRouter(
    prefix="/conversations",
    tags=["Conversations"],
)

conversation_repository = ConversationRepository()
conversation_message_repository = ConversationMessageRepository()
conversation_access_policy = KnowledgeBaseAccessPolicy(
    knowledge_base_repository=KnowledgeBaseRepository(),
    document_repository=DocumentRepository(),
)
conversation_service = ConversationService(
    conversation_repository=conversation_repository,
    message_repository=conversation_message_repository,
    access_policy=conversation_access_policy,
)


@router.post(
    "/",
    status_code=status.HTTP_201_CREATED,
    response_model=ConversationResponse,
)
def create_conversation(
    request: ConversationCreateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ConversationResponse:
    try:
        conversation = conversation_service.create(
            db=db,
            user=current_user,
            mode=request.mode,
            knowledge_base_id=request.knowledge_base_id,
            title=request.title,
        )
        return ConversationResponse.model_validate(conversation)
    except ResourceAccessNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


@router.get(
    "/",
    response_model=list[ConversationResponse],
)
def list_conversations(
    mode: ConversationMode | None = Query(default=None),
    knowledge_base_id: int | None = Query(default=None, gt=0),
    limit: int = Query(default=50, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[ConversationResponse]:
    conversations = conversation_service.list_owned(
        db=db,
        user_id=current_user.id,
        mode=mode,
        knowledge_base_id=knowledge_base_id,
        limit=limit,
    )
    return [
        ConversationResponse.model_validate(item)
        for item in conversations
    ]


@router.get(
    "/{conversation_id}",
    response_model=ConversationResponse,
)
def get_conversation(
    conversation_id: int = Path(gt=0),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ConversationResponse:
    try:
        conversation = conversation_service.get_owned(
            db=db,
            user_id=current_user.id,
            conversation_id=conversation_id,
        )
        return ConversationResponse.model_validate(conversation)
    except ConversationNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc


@router.get(
    "/{conversation_id}/messages",
    response_model=list[ConversationMessageResponse],
)
def list_conversation_messages(
    conversation_id: int = Path(gt=0),
    limit: int = Query(default=200, ge=1, le=500),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[ConversationMessageResponse]:
    try:
        messages = conversation_service.list_messages(
            db=db,
            user_id=current_user.id,
            conversation_id=conversation_id,
            limit=limit,
        )
        return [
            ConversationMessageResponse.model_validate(item)
            for item in messages
        ]
    except ConversationNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
