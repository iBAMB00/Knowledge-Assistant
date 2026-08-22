from sqlalchemy.orm import Session

from app.models.database.agent_checkpoint import AgentCheckpoint


class AgentCheckpointRepository:
    """AgentCheckpoint Repository：只负责 query / add / flush。"""

    def create(
        self,
        db: Session,
        checkpoint: AgentCheckpoint,
    ) -> AgentCheckpoint:
        db.add(checkpoint)
        db.flush()
        return checkpoint

    def find_latest_by_thread_id(
        self,
        db: Session,
        agent_thread_id: int,
    ) -> AgentCheckpoint | None:
        return (
            db.query(AgentCheckpoint)
            .filter(AgentCheckpoint.agent_thread_id == agent_thread_id)
            .order_by(AgentCheckpoint.sequence.desc())
            .first()
        )

    def list_by_thread_id(
        self,
        db: Session,
        agent_thread_id: int,
    ) -> list[AgentCheckpoint]:
        return (
            db.query(AgentCheckpoint)
            .filter(AgentCheckpoint.agent_thread_id == agent_thread_id)
            .order_by(AgentCheckpoint.sequence.asc())
            .all()
        )
