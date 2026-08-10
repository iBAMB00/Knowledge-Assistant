from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.dependencies.auth import get_current_user
from app.core.database import get_db
from app.models.database.user import User
from app.repositories.document_repository import DocumentRepository
from app.repositories.knowledge_base_repository import KnowledgeBaseRepository
from app.schemas.knowledge_base_create_request import KnowledgeBaseCreateRequest
from app.schemas.knowledge_base_response import KnowledgeBaseResponse
from app.schemas.knowledge_base_update_request import KnowledgeBaseUpdateRequest
from app.services.knowledge_base_access_policy import (
    KnowledgeBaseAccessPolicy,
    ResourceAccessNotFoundError,
)
from app.services.knowledge_base_service import (
    KnowledgeBaseConflictError,
    KnowledgeBaseService,
)

router = APIRouter(prefix="/knowledge-bases", tags=["Knowledge Bases"])

knowledge_base_repository = KnowledgeBaseRepository()
document_repository = DocumentRepository()
knowledge_base_access_policy = KnowledgeBaseAccessPolicy(
    knowledge_base_repository=knowledge_base_repository,
    document_repository=document_repository,
)
knowledge_base_service = KnowledgeBaseService(
    knowledge_base_repository=knowledge_base_repository,
    document_repository=document_repository,
    access_policy=knowledge_base_access_policy,
)


@router.post("/", status_code=status.HTTP_201_CREATED, response_model=KnowledgeBaseResponse)
def create_knowledge_base(
    request: KnowledgeBaseCreateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> KnowledgeBaseResponse:
    try:
        return knowledge_base_service.create(
            db=db,
            user=current_user,
            name=request.name,
            description=request.description,
        )
    except KnowledgeBaseConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/", response_model=list[KnowledgeBaseResponse])
def list_knowledge_bases(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[KnowledgeBaseResponse]:
    return knowledge_base_service.list_accessible(db=db, user=current_user)


@router.get("/{knowledge_base_id}", response_model=KnowledgeBaseResponse)
def get_knowledge_base(
    knowledge_base_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> KnowledgeBaseResponse:
    try:
        return knowledge_base_service.get_by_id(db, current_user, knowledge_base_id)
    except ResourceAccessNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.patch("/{knowledge_base_id}", response_model=KnowledgeBaseResponse)
def update_knowledge_base(
    knowledge_base_id: int,
    request: KnowledgeBaseUpdateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> KnowledgeBaseResponse:
    try:
        return knowledge_base_service.update(
            db=db,
            user=current_user,
            knowledge_base_id=knowledge_base_id,
            name=request.name,
            description=request.description,
            update_name="name" in request.model_fields_set,
            update_description="description" in request.model_fields_set,
        )
    except ResourceAccessNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except KnowledgeBaseConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/{knowledge_base_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_knowledge_base(
    knowledge_base_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> None:
    try:
        knowledge_base_service.delete(db, current_user, knowledge_base_id)
    except ResourceAccessNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except KnowledgeBaseConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
