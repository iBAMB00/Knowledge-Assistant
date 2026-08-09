from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.constants.user_role import UserRole
from app.models.database.knowledge_base import KnowledgeBase
from app.models.database.user import User
from app.repositories.document_repository import DocumentRepository
from app.repositories.knowledge_base_repository import KnowledgeBaseRepository
from app.services.knowledge_base_access_policy import KnowledgeBaseAccessPolicy


class KnowledgeBaseConflictError(ValueError):
    """知识库状态冲突。"""


class KnowledgeBaseService:
    """知识库生命周期与所有权业务服务。"""

    def __init__(
        self,
        knowledge_base_repository: KnowledgeBaseRepository,
        document_repository: DocumentRepository,
        access_policy: KnowledgeBaseAccessPolicy,
    ) -> None:
        self.knowledge_base_repository = knowledge_base_repository
        self.document_repository = document_repository
        self.access_policy = access_policy

    @staticmethod
    def _normalize_name(name: str) -> str:
        normalized = name.strip()
        if not normalized:
            raise ValueError("knowledge base name cannot be empty")
        return normalized

    @staticmethod
    def _normalize_description(description: str | None) -> str | None:
        if description is None:
            return None
        normalized = description.strip()
        return normalized or None

    def create(
        self,
        db: Session,
        user: User,
        name: str,
        description: str | None = None,
    ) -> KnowledgeBase:
        knowledge_base = KnowledgeBase(
            owner_id=user.id,
            name=self._normalize_name(name),
            description=self._normalize_description(description),
        )
        try:
            self.knowledge_base_repository.create(db, knowledge_base)
            db.commit()
            db.refresh(knowledge_base)
            return knowledge_base
        except IntegrityError as exc:
            db.rollback()
            raise KnowledgeBaseConflictError(
                "knowledge base name already exists"
            ) from exc
        except Exception:
            db.rollback()
            raise

    def list_accessible(self, db: Session, user: User) -> list[KnowledgeBase]:
        if user.role == UserRole.ADMIN.value:
            return self.knowledge_base_repository.find_all(db)
        return self.knowledge_base_repository.find_by_owner_id(db, user.id)

    def get_by_id(
        self,
        db: Session,
        user: User,
        knowledge_base_id: int,
    ) -> KnowledgeBase:
        return self.access_policy.get_accessible_knowledge_base(
            db=db,
            knowledge_base_id=knowledge_base_id,
            user=user,
        )

    def update(
        self,
        db: Session,
        user: User,
        knowledge_base_id: int,
        *,
        name: str | None = None,
        description: str | None = None,
        update_name: bool = False,
        update_description: bool = False,
    ) -> KnowledgeBase:
        knowledge_base = self.get_by_id(db, user, knowledge_base_id)

        if update_name:
            if name is None:
                raise ValueError("knowledge base name cannot be null")
            knowledge_base.name = self._normalize_name(name)
        if update_description:
            knowledge_base.description = self._normalize_description(description)

        try:
            db.flush()
            db.commit()
            db.refresh(knowledge_base)
            return knowledge_base
        except IntegrityError as exc:
            db.rollback()
            raise KnowledgeBaseConflictError(
                "knowledge base name already exists"
            ) from exc
        except Exception:
            db.rollback()
            raise

    def delete(
        self,
        db: Session,
        user: User,
        knowledge_base_id: int,
    ) -> None:
        knowledge_base = self.get_by_id(db, user, knowledge_base_id)
        if self.document_repository.count_by_knowledge_base_id(
            db=db,
            knowledge_base_id=knowledge_base_id,
        ) > 0:
            raise KnowledgeBaseConflictError(
                "knowledge base must be empty before deletion"
            )

        try:
            self.knowledge_base_repository.delete(db, knowledge_base)
            db.commit()
        except Exception:
            db.rollback()
            raise
