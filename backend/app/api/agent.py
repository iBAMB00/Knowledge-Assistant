import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.agent.context import ToolExecutionContext
from app.agent.native_agent import AgentLoopError, NativeAgentRunner
from app.api.dependencies.agent import (
    get_agent_access_policy,
    get_native_agent_runner,
)
from app.api.dependencies.auth import get_current_user
from app.constants.user_role import UserRole
from app.core.database import get_db
from app.core.request_context import get_request_id
from app.models.database.user import User
from app.schemas.agent_chat_request import AgentChatRequest
from app.schemas.agent_chat_response import AgentChatResponse
from app.services.knowledge_base_access_policy import (
    KnowledgeBaseAccessPolicy,
    ResourceAccessNotFoundError,
)


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/agent",
    tags=["Agent"],
)


@router.post(
    "/chat",
    response_model=AgentChatResponse,
)
def agent_chat(
    request: AgentChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    access_policy: KnowledgeBaseAccessPolicy = Depends(
        get_agent_access_policy
    ),
    agent_runner: NativeAgentRunner = Depends(
        get_native_agent_runner
    ),
) -> AgentChatResponse:
    """
    执行一次同步 Native Agent 问答。

    knowledge_base_id 虽来自客户端，但只有在服务端权限校验通过后，
    才会进入 ToolExecutionContext 成为本次 Agent Run 的可信资源范围。
    """

    request_id = get_request_id()

    try:
        access_policy.get_accessible_knowledge_base(
            db=db,
            knowledge_base_id=request.knowledge_base_id,
            user=current_user,
        )

        context = ToolExecutionContext(
            user_id=current_user.id,
            role=UserRole(current_user.role),
            knowledge_base_id=request.knowledge_base_id,
            request_id=request_id,
        )

        result = agent_runner.run(
            db=db,
            context=context,
            message=request.message,
        )

        return AgentChatResponse(answer=result.answer)

    except ResourceAccessNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except AgentLoopError as exc:
        logger.warning(
            "Agent chat stopped by runtime policy: request_id=%s "
            "error_code=%s",
            request_id,
            exc.code,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent暂时无法完成请求",
        ) from exc

    except Exception as exc:
        logger.error(
            "Agent chat failed: request_id=%s error_type=%s",
            request_id,
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Agent问答失败",
        ) from exc
