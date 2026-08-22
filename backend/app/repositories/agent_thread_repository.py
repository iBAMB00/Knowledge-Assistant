from sqlalchemy.orm import Session

from app.models.database.agent_thread import AgentThread


class AgentThreadRepository:
    """AgentThread Repository：只负责 query / add / flush。"""

    def create(self, db: Session, thread: AgentThread) -> AgentThread:
        db.add(thread)
        db.flush()
        return thread

    def find_by_thread_id(
        self,
        db: Session,
        thread_id: str,
    ) -> AgentThread | None:
        return (
            db.query(AgentThread)
            .filter(AgentThread.thread_id == thread_id)
            .first()
        )

    def find_by_thread_id_for_update(
        self,
        db: Session,
        thread_id: str,
    ) -> AgentThread | None:
        """写入生命周期状态前锁定 Thread，串行化 Cancel/Checkpoint 竞争。"""

        return (
            db.query(AgentThread)
            .filter(AgentThread.thread_id == thread_id)
            .with_for_update()
            .populate_existing()
            .first()
        )

    def find_by_conversation_id(
        self,
        db: Session,
        conversation_id: int,
    ) -> AgentThread | None:
        return (
            db.query(AgentThread)
            .filter(AgentThread.conversation_id == conversation_id)
            .first()
        )
