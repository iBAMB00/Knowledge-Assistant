from sqlalchemy.orm import Session

from app.models.database.agent_tool_call import AgentToolCall


class AgentToolCallRepository:
    """AgentToolCall 持久化仓库，只负责查询、add 与 flush。"""

    def create(
        self,
        db: Session,
        tool_call: AgentToolCall,
    ) -> AgentToolCall:
        db.add(tool_call)
        db.flush()
        return tool_call

    def find_by_id(
        self,
        db: Session,
        tool_call_id: int,
    ) -> AgentToolCall | None:
        return (
            db.query(AgentToolCall)
            .filter(AgentToolCall.id == tool_call_id)
            .first()
        )

    def find_all_by_agent_run_id(
        self,
        db: Session,
        agent_run_id: int,
    ) -> list[AgentToolCall]:
        return (
            db.query(AgentToolCall)
            .filter(AgentToolCall.agent_run_id == agent_run_id)
            .order_by(AgentToolCall.id.asc())
            .all()
        )
