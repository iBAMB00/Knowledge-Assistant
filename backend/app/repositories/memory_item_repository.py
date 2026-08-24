from sqlalchemy.orm import Session

from app.constants.memory_status import MemoryStatus
from app.models.database.memory_item import MemoryItem


class MemoryItemRepository:
    """MemoryItem 数据访问层；只负责查询、写入与 flush。"""

    def create(
        self,
        db: Session,
        memory: MemoryItem,
    ) -> MemoryItem:
        db.add(memory)
        db.flush()
        return memory

    def find_by_id_and_conversation(
        self,
        db: Session,
        *,
        memory_id: int,
        source_conversation_id: int,
    ) -> MemoryItem | None:
        return (
            db.query(MemoryItem)
            .filter(
                MemoryItem.id == memory_id,
                MemoryItem.source_conversation_id == source_conversation_id,
            )
            .one_or_none()
        )

    def find_active_by_conversation(
        self,
        db: Session,
        *,
        source_conversation_id: int,
        limit: int = 100,
    ) -> list[MemoryItem]:
        return (
            db.query(MemoryItem)
            .filter(
                MemoryItem.source_conversation_id == source_conversation_id,
                MemoryItem.status == MemoryStatus.ACTIVE.value,
            )
            .order_by(MemoryItem.id.asc())
            .limit(limit)
            .all()
        )

    def update_status(
        self,
        db: Session,
        *,
        memory: MemoryItem,
        status: MemoryStatus,
    ) -> MemoryItem:
        memory.status = status.value
        db.flush()
        return memory
