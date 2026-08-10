from sqlalchemy.orm import Session

from app.constants.user_role import UserRole
from app.models.database.document import Document
from app.models.database.knowledge_base import KnowledgeBase
from app.models.database.user import User
from app.repositories.document_repository import DocumentRepository
from app.repositories.knowledge_base_repository import KnowledgeBaseRepository


class ResourceAccessNotFoundError(ValueError):
    """资源不存在或当前用户无权知道该资源存在。"""


class KnowledgeBaseAccessPolicy:
    """集中处理 KnowledgeBase / Document 的最小资源授权。"""

    def __init__(
        self,
        knowledge_base_repository: KnowledgeBaseRepository,
        document_repository: DocumentRepository,
    ) -> None:
        self.knowledge_base_repository = knowledge_base_repository
        self.document_repository = document_repository

    @staticmethod
    def _is_admin(user: User) -> bool:
        return user.role == UserRole.ADMIN.value

    def get_accessible_knowledge_base(
        self,
        db: Session,
        knowledge_base_id: int,
        user: User,
    ) -> KnowledgeBase:
        knowledge_base = self.knowledge_base_repository.find_by_id(
            db=db,
            knowledge_base_id=knowledge_base_id,
        )
        if knowledge_base is None or (
            knowledge_base.owner_id != user.id
            and not self._is_admin(user)
        ):
            raise ResourceAccessNotFoundError("knowledge base not found")
        return knowledge_base

    def get_accessible_document(
        self,
        db: Session,
        document_id: int,
        user: User,
    ) -> Document:
        document = self.document_repository.find_by_id(
            db=db,
            document_id=document_id,
        )
        if document is None or document.knowledge_base_id is None:
            raise ResourceAccessNotFoundError("document not found")

        try:
            self.get_accessible_knowledge_base(
                db=db,
                knowledge_base_id=document.knowledge_base_id,
                user=user,
            )
        except ResourceAccessNotFoundError as exc:
            raise ResourceAccessNotFoundError("document not found") from exc
        return document

    def ensure_document_in_knowledge_base(
        self,
        db: Session,
        document_id: int,
        knowledge_base_id: int,
        user: User,
    ) -> Document:
        document = self.get_accessible_document(
            db=db,
            document_id=document_id,
            user=user,
        )
        if document.knowledge_base_id != knowledge_base_id:
            raise ResourceAccessNotFoundError("document not found")
        return document
