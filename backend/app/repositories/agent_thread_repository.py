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
