from sqlalchemy.orm import Session

from app.models.database.agent_run import AgentRun


class AgentRunRepository:
    """AgentRun 持久化仓库，只负责查询、add 与 flush。"""

    def create(
        self,
        db: Session,
        agent_run: AgentRun,
    ) -> AgentRun:
        db.add(agent_run)
        db.flush()
        return agent_run

    def find_by_id(
        self,
        db: Session,
        agent_run_id: int,
    ) -> AgentRun | None:
        return (
            db.query(AgentRun)
            .filter(AgentRun.id == agent_run_id)
            .first()
        )

    def find_latest_by_request_id(
        self,
        db: Session,
        request_id: str,
    ) -> AgentRun | None:
        return (
            db.query(AgentRun)
            .filter(AgentRun.request_id == request_id)
            .order_by(AgentRun.id.desc())
            .first()
        )

    def find_by_id_and_user_id(
        self,
        db: Session,
        agent_run_id: int,
        user_id: int,
    ) -> AgentRun | None:
        return (
            db.query(AgentRun)
            .filter(
                AgentRun.id == agent_run_id,
                AgentRun.user_id == user_id,
            )
            .first()
        )

    def find_recent_by_user_id(
        self,
        db: Session,
        user_id: int,
        *,
        knowledge_base_id: int | None = None,
        limit: int = 20,
    ) -> list[AgentRun]:
        query = db.query(AgentRun).filter(
            AgentRun.user_id == user_id
        )

        if knowledge_base_id is not None:
            query = query.filter(
                AgentRun.knowledge_base_id == knowledge_base_id
            )

        return (
            query
            .order_by(AgentRun.id.desc())
            .limit(limit)
            .all()
        )
