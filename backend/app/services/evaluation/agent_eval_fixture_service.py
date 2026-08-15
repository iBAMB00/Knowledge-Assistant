from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.constants.document_status import DocumentStatus
from app.constants.user_role import UserRole
from app.models.database.document import Document
from app.models.database.knowledge_base import KnowledgeBase
from app.models.database.processing_job import ProcessingJob
from app.models.database.user import User
from app.repositories.document_repository import DocumentRepository
from app.repositories.knowledge_base_repository import KnowledgeBaseRepository
from app.repositories.user_repository import UserRepository
from app.schemas.agent_evaluation import AgentEvaluationFixtureManifest


class AgentEvaluationFixtureService:
    """准备 D2.5 Live Eval 所需的最小、可重复数据库 Fixture。"""

    PRIMARY_EMAIL = "agent-eval-primary@fixture.invalid"
    CROSS_USER_EMAIL = "agent-eval-cross-user@fixture.invalid"
    PRIMARY_KB_NAME = "__agent_eval_primary__"
    CROSS_USER_KB_NAME = "__agent_eval_cross_user__"
    PRIMARY_DOCUMENT_FILENAME = "__agent_eval_primary_document__.txt"
    CROSS_USER_DOCUMENT_FILENAME = "__agent_eval_cross_user_document__.txt"
    PRIMARY_STORAGE_KEY = "agent-eval-fixture/primary/document.txt"
    CROSS_USER_STORAGE_KEY = "agent-eval-fixture/cross-user/document.txt"
    NON_LOGIN_PASSWORD_HASH = "agent-eval-fixture-no-login"
    MISSING_JOB_ID_START = 2_000_000_000

    def __init__(
        self,
        *,
        user_repository: UserRepository,
        knowledge_base_repository: KnowledgeBaseRepository,
        document_repository: DocumentRepository,
    ) -> None:
        self.user_repository = user_repository
        self.knowledge_base_repository = knowledge_base_repository
        self.document_repository = document_repository

    def prepare(self, *, db: Session) -> AgentEvaluationFixtureManifest:
        """
        创建或复用 Live Eval Fixture，并一次性提交事务。

        Fixture 用户使用不可登录占位 hash，只作为可信执行身份，不用于认证。
        文档只准备元数据；D2.5 不负责真实检索语料和向量索引。
        """

        try:
            primary_user = self._get_or_create_user(
                db=db,
                email=self.PRIMARY_EMAIL,
            )
            cross_user = self._get_or_create_user(
                db=db,
                email=self.CROSS_USER_EMAIL,
            )
            primary_kb = self._get_or_create_knowledge_base(
                db=db,
                owner=primary_user,
                name=self.PRIMARY_KB_NAME,
                description="Agent Live Eval primary scope",
            )
            cross_user_kb = self._get_or_create_knowledge_base(
                db=db,
                owner=cross_user,
                name=self.CROSS_USER_KB_NAME,
                description="Agent Live Eval cross-user scope",
            )
            primary_document = self._get_or_create_document(
                db=db,
                knowledge_base=primary_kb,
                filename=self.PRIMARY_DOCUMENT_FILENAME,
                storage_key=self.PRIMARY_STORAGE_KEY,
            )
            cross_user_document = self._get_or_create_document(
                db=db,
                knowledge_base=cross_user_kb,
                filename=self.CROSS_USER_DOCUMENT_FILENAME,
                storage_key=self.CROSS_USER_STORAGE_KEY,
            )
            missing_job_id = self._find_missing_processing_job_id(db=db)

            db.commit()

            for entity in (
                primary_user,
                cross_user,
                primary_kb,
                cross_user_kb,
                primary_document,
                cross_user_document,
            ):
                db.refresh(entity)

            return AgentEvaluationFixtureManifest(
                schema_version="1.0",
                fixture_version="1.0.0",
                generated_at=datetime.now(timezone.utc),
                primary_user_id=primary_user.id,
                primary_role=UserRole.USER.value,
                primary_knowledge_base_id=primary_kb.id,
                primary_document_id=primary_document.id,
                cross_user_id=cross_user.id,
                cross_user_knowledge_base_id=cross_user_kb.id,
                cross_user_document_id=cross_user_document.id,
                missing_processing_job_id=missing_job_id,
                bindings={
                    "primary_document_id": primary_document.id,
                    "cross_user_document_id": cross_user_document.id,
                    "missing_processing_job_id": missing_job_id,
                },
            )
        except Exception:
            db.rollback()
            raise

    def _get_or_create_user(
        self,
        *,
        db: Session,
        email: str,
    ) -> User:
        existing = self.user_repository.find_by_email(db=db, email=email)
        if existing is not None:
            if existing.role != UserRole.USER.value or not existing.is_active:
                raise ValueError(
                    "agent evaluation fixture user has unexpected state: "
                    f"{email}"
                )
            return existing

        return self.user_repository.create(
            db=db,
            user=User(
                email=email,
                password_hash=self.NON_LOGIN_PASSWORD_HASH,
                role=UserRole.USER.value,
                is_active=True,
            ),
        )

    def _get_or_create_knowledge_base(
        self,
        *,
        db: Session,
        owner: User,
        name: str,
        description: str,
    ) -> KnowledgeBase:
        existing = next(
            (
                knowledge_base
                for knowledge_base in self.knowledge_base_repository.find_by_owner_id(
                    db=db,
                    owner_id=owner.id,
                )
                if knowledge_base.name == name
            ),
            None,
        )
        if existing is not None:
            return existing

        return self.knowledge_base_repository.create(
            db=db,
            knowledge_base=KnowledgeBase(
                owner_id=owner.id,
                name=name,
                description=description,
            ),
        )

    def _get_or_create_document(
        self,
        *,
        db: Session,
        knowledge_base: KnowledgeBase,
        filename: str,
        storage_key: str,
    ) -> Document:
        existing = next(
            (
                document
                for document in self.document_repository.find_all(
                    db=db,
                    knowledge_base_id=knowledge_base.id,
                )
                if document.filename == filename
            ),
            None,
        )
        if existing is not None:
            if existing.storage_key != storage_key:
                raise ValueError(
                    "agent evaluation fixture document has unexpected storage_key"
                )
            return existing

        return self.document_repository.create(
            db=db,
            document=Document(
                knowledge_base_id=knowledge_base.id,
                filename=filename,
                storage_key=storage_key,
                size=len(filename.encode("utf-8")),
                status=DocumentStatus.COMPLETED.value,
            ),
        )

    @classmethod
    def _find_missing_processing_job_id(cls, *, db: Session) -> int:
        candidate = cls.MISSING_JOB_ID_START
        while candidate > 0:
            if db.get(ProcessingJob, candidate) is None:
                return candidate
            candidate -= 1
        raise RuntimeError("cannot allocate missing processing job id")
