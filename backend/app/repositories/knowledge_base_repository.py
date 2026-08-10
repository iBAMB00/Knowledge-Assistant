from sqlalchemy.orm import Session

from app.models.database.knowledge_base import KnowledgeBase


class KnowledgeBaseRepository:
    """KnowledgeBase 数据访问层；只查询、写入和 flush。"""

    def create(self, db: Session, knowledge_base: KnowledgeBase) -> KnowledgeBase:
        db.add(knowledge_base)
        db.flush()
        return knowledge_base

    def find_by_id(self, db: Session, knowledge_base_id: int) -> KnowledgeBase | None:
        return db.get(KnowledgeBase, knowledge_base_id)

    def find_by_owner_id(self, db: Session, owner_id: int) -> list[KnowledgeBase]:
        return (
            db.query(KnowledgeBase)
            .filter(KnowledgeBase.owner_id == owner_id)
            .order_by(KnowledgeBase.created_at.desc(), KnowledgeBase.id.desc())
            .all()
        )

    def find_all(self, db: Session) -> list[KnowledgeBase]:
        return (
            db.query(KnowledgeBase)
            .order_by(KnowledgeBase.created_at.desc(), KnowledgeBase.id.desc())
            .all()
        )

    def delete(self, db: Session, knowledge_base: KnowledgeBase) -> None:
        db.delete(knowledge_base)
        db.flush()
