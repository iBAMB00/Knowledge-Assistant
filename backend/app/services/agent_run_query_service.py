from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.models.database.agent_run import AgentRun
from app.models.database.agent_tool_call import AgentToolCall
from app.repositories.agent_run_repository import AgentRunRepository
from app.repositories.agent_tool_call_repository import AgentToolCallRepository


class AgentRunNotFoundError(ValueError):
    """AgentRun 不存在，或当前用户无权知道该 Run 存在。"""


@dataclass(frozen=True)
class AgentRunDetail:
    """AgentRun 与其 ToolCall 的只读查询结果。"""

    run: AgentRun
    tool_calls: tuple[AgentToolCall, ...]


class AgentRunQueryService:
    """
    AgentRun 只读查询服务。

    C2 默认只允许用户查询自己发起的 AgentRun；即使角色为 admin，
    也不会通过普通用户接口自动获得跨用户运行记录访问权。
    """

    def __init__(
        self,
        *,
        agent_run_repository: AgentRunRepository,
        tool_call_repository: AgentToolCallRepository,
    ) -> None:
        self.agent_run_repository = agent_run_repository
        self.tool_call_repository = tool_call_repository

    def list_owned_runs(
        self,
        *,
        db: Session,
        user_id: int,
        knowledge_base_id: int | None = None,
        limit: int = 20,
    ) -> list[AgentRun]:
        """按时间倒序返回当前用户自己的最近 AgentRun。"""

        if user_id <= 0:
            raise ValueError("user_id must be positive")
        if knowledge_base_id is not None and knowledge_base_id <= 0:
            raise ValueError("knowledge_base_id must be positive")
        if limit < 1 or limit > 100:
            raise ValueError("limit must be between 1 and 100")

        return self.agent_run_repository.find_recent_by_user_id(
            db=db,
            user_id=user_id,
            knowledge_base_id=knowledge_base_id,
            limit=limit,
        )

    def get_owned_run_detail(
        self,
        *,
        db: Session,
        user_id: int,
        agent_run_id: int,
    ) -> AgentRunDetail:
        """返回当前用户自己的单次 AgentRun 及 ToolCall 摘要。"""

        if user_id <= 0:
            raise ValueError("user_id must be positive")
        if agent_run_id <= 0:
            raise ValueError("agent_run_id must be positive")

        agent_run = self.agent_run_repository.find_by_id_and_user_id(
            db=db,
            agent_run_id=agent_run_id,
            user_id=user_id,
        )
        if agent_run is None:
            raise AgentRunNotFoundError("agent run not found")

        tool_calls = self.tool_call_repository.find_all_by_agent_run_id(
            db=db,
            agent_run_id=agent_run.id,
        )

        return AgentRunDetail(
            run=agent_run,
            tool_calls=tuple(tool_calls),
        )
