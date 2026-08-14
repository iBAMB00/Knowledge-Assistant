from collections.abc import Iterator
import json
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.agent.context import ToolExecutionContext
from app.agent.native_agent import AgentLoopError, NativeAgentRunner
from app.agent.run_event import AgentRunEvent
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
        context = _build_authorized_context(
            db=db,
            request=request,
            current_user=current_user,
            access_policy=access_policy,
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


@router.post("/chat/stream")
def stream_agent_chat(
    request: AgentChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    access_policy: KnowledgeBaseAccessPolicy = Depends(
        get_agent_access_policy
    ),
    agent_runner: NativeAgentRunner = Depends(
        get_native_agent_runner
    ),
) -> StreamingResponse:
    """
    以 SSE 输出 Native Agent 的安全运行事件。

    当前 B5 流式的是 Agent 生命周期事件；最终回答仍由一次完整模型
    响应产生，不把完整答案人为切片伪装成 token streaming。
    """

    request_id = get_request_id()

    try:
        context = _build_authorized_context(
            db=db,
            request=request,
            current_user=current_user,
            access_policy=access_policy,
            request_id=request_id,
        )

        # StreamingResponse 一旦返回便已进入 200 响应，
        # 因此基础输入错误必须在开始流之前完成校验。
        message = request.message.strip()
        if not message:
            raise ValueError("message cannot be empty")

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

    except Exception as exc:
        logger.error(
            "Agent stream preparation failed: request_id=%s error_type=%s",
            request_id,
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Agent问答准备失败",
        ) from exc

    return StreamingResponse(
        generate_agent_chat_sse(
            db=db,
            context=context,
            message=message,
            agent_runner=agent_runner,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def generate_agent_chat_sse(
    *,
    db: Session,
    context: ToolExecutionContext,
    message: str,
    agent_runner: NativeAgentRunner,
) -> Iterator[str]:
    """
    把 provider-neutral AgentRunEvent 编码成 SSE。

    正常结束发送 done；流中失败发送 error；客户端取消时向下关闭
    Runtime 事件生成器。不会输出隐藏推理、Tool 参数或 Tool Result 正文。
    """

    event_stream: Iterator[AgentRunEvent] | None = None

    try:
        event_stream = agent_runner.run_events(
            db=db,
            context=context,
            message=message,
        )

        for event in event_stream:
            yield _encode_agent_sse_event(event)

        yield "event: done\ndata: {}\n\n"

    except GeneratorExit:
        logger.info(
            "Agent SSE cancelled: request_id=%s",
            context.request_id,
        )
        raise

    except AgentLoopError as exc:
        logger.warning(
            "Agent SSE stopped by runtime policy: request_id=%s "
            "error_code=%s",
            context.request_id,
            exc.code,
        )
        yield _encode_sse(
            "error",
            {"message": "Agent暂时无法完成请求"},
        )

    except Exception as exc:
        logger.error(
            "Agent SSE failed: request_id=%s error_type=%s",
            context.request_id,
            type(exc).__name__,
        )
        yield _encode_sse(
            "error",
            {"message": "Agent问答失败"},
        )

    finally:
        _close_iterator(event_stream)


def _build_authorized_context(
    *,
    db: Session,
    request: AgentChatRequest,
    current_user: User,
    access_policy: KnowledgeBaseAccessPolicy,
    request_id: str,
) -> ToolExecutionContext:
    """先校验客户端 KB Scope，再固化为服务端可信执行上下文。"""

    access_policy.get_accessible_knowledge_base(
        db=db,
        knowledge_base_id=request.knowledge_base_id,
        user=current_user,
    )

    return ToolExecutionContext(
        user_id=current_user.id,
        role=UserRole(current_user.role),
        knowledge_base_id=request.knowledge_base_id,
        request_id=request_id,
    )


def _encode_agent_sse_event(event: AgentRunEvent) -> str:
    """把 Runtime 事件转换成对外最小、安全的 SSE Payload。"""

    if event.type == "message":
        payload: dict[str, Any] = {
            "content": event.content,
        }
    else:
        payload = event.model_dump(
            exclude={"type"},
            exclude_none=True,
        )

    return _encode_sse(event.type, payload)


def _encode_sse(event_name: str, payload: dict[str, Any]) -> str:
    """统一编码单个 SSE 事件。"""

    data = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    return (
        f"event: {event_name}\n"
        f"data: {data}\n\n"
    )


def _close_iterator(iterator: Any | None) -> None:
    """客户端取消或异常时尽量关闭底层 Runtime 生成器。"""

    if iterator is None:
        return

    close_method = getattr(iterator, "close", None)
    if not callable(close_method):
        return

    try:
        close_method()
    except Exception as exc:
        logger.warning(
            "Failed to close agent event stream: request_id=unknown "
            "error_type=%s",
            type(exc).__name__,
        )
