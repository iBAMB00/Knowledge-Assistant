from collections.abc import Iterator
import json
import logging
from typing import Any, cast

from fastapi import APIRouter, Depends, HTTPException, Path as ApiPath, Query, Response, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.agent.context import ToolExecutionContext
from app.agent.frameworks.langchain.runner import LangChainAgentError
from app.agent.hitl import (
    AgentApprovalSelection,
    AgentApprovalStateError,
    AgentInterruptRequired,
)
from app.agent.checkpoint import (
    AgentResumeCheckpointNotFoundError,
    AgentResumeStateError,
)
from app.agent.run_control import (
    AgentRunCancellationError,
    AgentRunControlNotFoundError,
    AgentRunStateError,
)
from app.agent.native_agent import AgentLoopError
from app.agent.run_event import AgentRunEvent
from app.api.dependencies.agent import (
    get_agent_access_policy,
    get_agent_run_query_service,
    get_agent_runtime_diagnostics_service,
    get_agent_runtime_selector,
    get_agent_hitl_service,
    get_agent_run_control_service,
    get_agent_thread_status_service,
    get_conversation_memory_extraction_service,
)
from app.api.dependencies.auth import get_current_user
from app.api.dependencies.conversation import get_conversation_service
from app.constants.agent_runtime import AgentRuntime
from app.constants.conversation_message_role import ConversationMessageRole
from app.constants.conversation_mode import ConversationMode
from app.constants.user_role import UserRole
from app.core.database import get_db
from app.core.request_context import get_request_id
from app.models.database.conversation_message import ConversationMessage
from app.models.database.user import User
from app.schemas.agent_chat_request import AgentChatRequest
from app.schemas.agent_chat_response import AgentChatResponse
from app.schemas.agent_run_response import (
    AgentRunDetailResponse,
    AgentRunSummaryResponse,
    AgentToolCallSummaryResponse,
)
from app.schemas.agent_runtime_status import AgentRuntimeStatusResponse
from app.schemas.agent_stateful_control import (
    AgentApprovalRequirementResponse,
    AgentThreadActionRequest,
    AgentThreadApprovalRequest,
    AgentThreadStatusResponse,
    AgentWaitingResponse,
)
from app.services.agent_runtime_diagnostics_service import (
    AgentRuntimeDiagnosticsService,
)
from app.services.agent_hitl_service import AgentHITLService
from app.services.agent_run_control_service import AgentRunControlService
from app.services.agent_thread_status_service import (
    AgentThreadNotFoundError,
    AgentThreadStatusService,
    AgentThreadStatusSnapshot,
)
from app.services.langgraph_agent_execution_service import (
    AgentStatefulRunConflictError,
    LangGraphAgentExecutionService,
)
from app.services.agent_runtime_selector import (
    AgentRuntimeExecutionService,
    AgentRuntimeSelector,
    AgentRuntimeUnavailableError,
)
from app.services.agent_run_query_service import (
    AgentRunNotFoundError,
    AgentRunQueryService,
)
from app.services.conversation_service import (
    ConversationNotFoundError,
    ConversationScopeConflictError,
    ConversationService,
)
from app.services.conversation_memory_extraction_service import (
    ConversationMemoryExtractionError,
    ConversationMemoryExtractionService,
)
from app.services.knowledge_base_access_policy import (
    KnowledgeBaseAccessPolicy,
    ResourceAccessNotFoundError,
)


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/agent",
    tags=["Agent"],
)




@router.get(
    "/runtimes",
    response_model=AgentRuntimeStatusResponse,
)
def get_agent_runtimes(
    _current_user: User = Depends(get_current_user),
    diagnostics_service: AgentRuntimeDiagnosticsService = Depends(
        get_agent_runtime_diagnostics_service
    ),
) -> AgentRuntimeStatusResponse:
    """返回当前部署可选择的 Agent Runtime 与安全能力摘要。"""

    return diagnostics_service.get_runtime_status()


@router.get(
    "/runs",
    response_model=list[AgentRunSummaryResponse],
)
def list_agent_runs(
    knowledge_base_id: int | None = Query(
        default=None,
        gt=0,
    ),
    limit: int = Query(
        default=20,
        ge=1,
        le=100,
    ),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    access_policy: KnowledgeBaseAccessPolicy = Depends(
        get_agent_access_policy
    ),
    query_service: AgentRunQueryService = Depends(
        get_agent_run_query_service
    ),
) -> list[AgentRunSummaryResponse]:
    """查询当前用户自己的最近 AgentRun 摘要。"""

    try:
        if knowledge_base_id is not None:
            access_policy.get_accessible_knowledge_base(
                db=db,
                knowledge_base_id=knowledge_base_id,
                user=current_user,
            )

        runs = query_service.list_owned_runs(
            db=db,
            user_id=current_user.id,
            knowledge_base_id=knowledge_base_id,
            limit=limit,
        )
        return [
            AgentRunSummaryResponse.model_validate(run)
            for run in runs
        ]

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


@router.get(
    "/runs/{agent_run_id}",
    response_model=AgentRunDetailResponse,
)
def get_agent_run(
    agent_run_id: int = ApiPath(gt=0),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    query_service: AgentRunQueryService = Depends(
        get_agent_run_query_service
    ),
) -> AgentRunDetailResponse:
    """查询当前用户自己的 AgentRun 详情与 ToolCall 摘要。"""

    try:
        detail = query_service.get_owned_run_detail(
            db=db,
            user_id=current_user.id,
            agent_run_id=agent_run_id,
        )

        run = detail.run
        return AgentRunDetailResponse(
            id=run.id,
            knowledge_base_id=run.knowledge_base_id,
            status=run.status,
            tool_call_count=run.tool_call_count,
            started_at=run.started_at,
            finished_at=run.finished_at,
            request_id=run.request_id,
            model_provider=run.model_provider,
            model_name=run.model_name,
            agent_version=run.agent_version,
            prompt_version=run.prompt_version,
            toolset_version=run.toolset_version,
            retrieval_config_version=run.retrieval_config_version,
            eval_dataset_version=run.eval_dataset_version,
            evaluator_version=run.evaluator_version,
            error_type=run.error_type,
            tool_calls=[
                AgentToolCallSummaryResponse.model_validate(tool_call)
                for tool_call in detail.tool_calls
            ],
        )

    except AgentRunNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


@router.post(
    "/chat",
    response_model=AgentChatResponse | AgentWaitingResponse,
)
def agent_chat(
    request: AgentChatRequest,
    response: Response,
    runtime: AgentRuntime = Query(default=AgentRuntime.NATIVE),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    access_policy: KnowledgeBaseAccessPolicy = Depends(
        get_agent_access_policy
    ),
    runtime_selector: AgentRuntimeSelector = Depends(
        get_agent_runtime_selector
    ),
    conversation_service: ConversationService = Depends(
        get_conversation_service
    ),
) -> AgentChatResponse | AgentWaitingResponse:
    """
    执行一次同步 Agent 问答；默认 Native，可显式选择已开放的 Candidate。

    knowledge_base_id 虽来自客户端，但只有在服务端权限校验通过后，
    才会进入 ToolExecutionContext 成为本次 Agent Run 的可信资源范围。
    Runtime 选择只改变编排实现，不改变身份、KB Scope、Tool 或安全边界。
    """

    request_id = get_request_id()

    try:
        conversation_id = _prepare_conversation_persistence(
            db=db,
            conversation_service=conversation_service,
            current_user=current_user,
            conversation_id=request.conversation_id,
            knowledge_base_id=request.knowledge_base_id,
            runtime=runtime,
        )
        context = _build_authorized_context(
            db=db,
            request=request,
            current_user=current_user,
            access_policy=access_policy,
            request_id=request_id,
            conversation_id=conversation_id,
        )

        agent_runner = runtime_selector.select(runtime)
        user_message = _append_conversation_message(
            db=db,
            conversation_service=conversation_service,
            user_id=current_user.id,
            conversation_id=conversation_id,
            role=ConversationMessageRole.USER,
            content=request.message,
        )

        result = agent_runner.run(
            db=db,
            context=context,
            message=request.message,
        )

        assistant_message = _append_conversation_message(
            db=db,
            conversation_service=conversation_service,
            user_id=current_user.id,
            conversation_id=conversation_id,
            role=ConversationMessageRole.ASSISTANT,
            content=result.answer,
        )
        _extract_completed_turn_memory(
            db=db,
            memory_extraction_service=None,
            user_id=current_user.id,
            knowledge_base_id=request.knowledge_base_id,
            conversation_id=conversation_id,
            user_message=user_message,
            assistant_message=assistant_message,
        )

        return AgentChatResponse(answer=result.answer)

    except AgentInterruptRequired as exc:
        response.status_code = status.HTTP_202_ACCEPTED
        return _build_waiting_response(exc)

    except AgentStatefulRunConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    except AgentRunCancellationError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Agent任务已取消",
        ) from exc

    except (ResourceAccessNotFoundError, ConversationNotFoundError) as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    except ConversationScopeConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except AgentRuntimeUnavailableError as exc:
        logger.warning(
            "Agent runtime unavailable: request_id=%s runtime=%s",
            request_id,
            runtime.value,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent运行时暂不可用",
        ) from exc

    except (AgentLoopError, LangChainAgentError) as exc:
        logger.warning(
            "Agent chat stopped by runtime policy: request_id=%s "
            "runtime=%s error_code=%s",
            request_id,
            runtime.value,
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
    runtime: AgentRuntime = Query(default=AgentRuntime.NATIVE),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    access_policy: KnowledgeBaseAccessPolicy = Depends(
        get_agent_access_policy
    ),
    runtime_selector: AgentRuntimeSelector = Depends(
        get_agent_runtime_selector
    ),
    conversation_service: ConversationService = Depends(
        get_conversation_service
    ),
) -> StreamingResponse:
    """
    以 SSE 输出 Native / LangChain Candidate 的安全运行事件。

    两条 Runtime 统一输出 provider-neutral 生命周期事件；最终回答仍是
    完整 message event，不消费 reasoning/token stream，也不伪装 token streaming。
    """

    request_id = get_request_id()

    try:
        conversation_id = _prepare_conversation_persistence(
            db=db,
            conversation_service=conversation_service,
            current_user=current_user,
            conversation_id=request.conversation_id,
            knowledge_base_id=request.knowledge_base_id,
            runtime=runtime,
        )
        context = _build_authorized_context(
            db=db,
            request=request,
            current_user=current_user,
            access_policy=access_policy,
            request_id=request_id,
            conversation_id=conversation_id,
        )

        # StreamingResponse 一旦返回便已进入 200 响应，
        # 因此基础输入错误必须在开始流之前完成校验。
        message = request.message.strip()
        if not message:
            raise ValueError("message cannot be empty")

        agent_runner = runtime_selector.select(runtime)
        user_message = _append_conversation_message(
            db=db,
            conversation_service=conversation_service,
            user_id=current_user.id,
            conversation_id=conversation_id,
            role=ConversationMessageRole.USER,
            content=message,
        )

    except AgentStatefulRunConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    except (ResourceAccessNotFoundError, ConversationNotFoundError) as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    except ConversationScopeConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    except AgentRuntimeUnavailableError as exc:
        logger.warning(
            "Agent stream runtime unavailable: request_id=%s runtime=%s",
            request_id,
            runtime.value,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent运行时暂不可用",
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
            conversation_service=conversation_service,
            conversation_id=conversation_id,
            memory_extraction_service=None,
            source_user_message_id=(
                user_message.id if user_message is not None else None
            ),
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )



@router.get(
    "/threads/{thread_id}",
    response_model=AgentThreadStatusResponse,
)
def get_agent_thread_status(
    thread_id: str = ApiPath(min_length=1, max_length=128),
    knowledge_base_id: int = Query(gt=0),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    access_policy: KnowledgeBaseAccessPolicy = Depends(
        get_agent_access_policy
    ),
    status_service: AgentThreadStatusService = Depends(
        get_agent_thread_status_service
    ),
) -> AgentThreadStatusResponse:
    """查询当前用户自己的 Stateful Thread 安全状态。"""

    try:
        _ensure_kb_access(
            db=db,
            user=current_user,
            knowledge_base_id=knowledge_base_id,
            access_policy=access_policy,
        )
        snapshot = status_service.get_status(
            db,
            thread_id=thread_id,
            user_id=current_user.id,
            knowledge_base_id=knowledge_base_id,
        )
        return _build_thread_status_response(snapshot)
    except (ResourceAccessNotFoundError, AgentThreadNotFoundError) as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc


@router.post(
    "/threads/{thread_id}/approve",
    response_model=AgentThreadStatusResponse,
)
def approve_agent_thread(
    request: AgentThreadApprovalRequest,
    thread_id: str = ApiPath(min_length=1, max_length=128),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    access_policy: KnowledgeBaseAccessPolicy = Depends(
        get_agent_access_policy
    ),
    hitl_service: AgentHITLService = Depends(get_agent_hitl_service),
    status_service: AgentThreadStatusService = Depends(
        get_agent_thread_status_service
    ),
) -> AgentThreadStatusResponse:
    """Durable 批准待确认 ToolCall；批准本身不自动开始执行。"""

    try:
        _ensure_kb_access(
            db=db,
            user=current_user,
            knowledge_base_id=request.knowledge_base_id,
            access_policy=access_policy,
        )
        status_service.get_status(
            db,
            thread_id=thread_id,
            user_id=current_user.id,
            knowledge_base_id=request.knowledge_base_id,
        )
        hitl_service.approve(
            db,
            thread_id=thread_id,
            user_id=current_user.id,
            knowledge_base_id=request.knowledge_base_id,
            selection=AgentApprovalSelection(
                call_ids=tuple(request.call_ids)
            ),
        )
        snapshot = status_service.get_status(
            db,
            thread_id=thread_id,
            user_id=current_user.id,
            knowledge_base_id=request.knowledge_base_id,
        )
        return _build_thread_status_response(snapshot)
    except (ResourceAccessNotFoundError, AgentThreadNotFoundError) as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except AgentApprovalStateError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc


@router.post(
    "/threads/{thread_id}/reject",
    response_model=AgentThreadStatusResponse,
)
def reject_agent_thread(
    request: AgentThreadApprovalRequest,
    thread_id: str = ApiPath(min_length=1, max_length=128),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    access_policy: KnowledgeBaseAccessPolicy = Depends(
        get_agent_access_policy
    ),
    hitl_service: AgentHITLService = Depends(get_agent_hitl_service),
    status_service: AgentThreadStatusService = Depends(
        get_agent_thread_status_service
    ),
) -> AgentThreadStatusResponse:
    """拒绝 pending ToolCall 并把 Thread durable 终止为 CANCELLED。"""

    try:
        _ensure_kb_access(
            db=db,
            user=current_user,
            knowledge_base_id=request.knowledge_base_id,
            access_policy=access_policy,
        )
        status_service.get_status(
            db,
            thread_id=thread_id,
            user_id=current_user.id,
            knowledge_base_id=request.knowledge_base_id,
        )
        hitl_service.reject(
            db,
            thread_id=thread_id,
            user_id=current_user.id,
            knowledge_base_id=request.knowledge_base_id,
            selection=AgentApprovalSelection(
                call_ids=tuple(request.call_ids)
            ),
        )
        snapshot = status_service.get_status(
            db,
            thread_id=thread_id,
            user_id=current_user.id,
            knowledge_base_id=request.knowledge_base_id,
        )
        return _build_thread_status_response(snapshot)
    except (ResourceAccessNotFoundError, AgentThreadNotFoundError) as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except AgentApprovalStateError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc


@router.post(
    "/threads/{thread_id}/cancel",
    response_model=AgentThreadStatusResponse,
)
def cancel_agent_thread(
    request: AgentThreadActionRequest,
    thread_id: str = ApiPath(min_length=1, max_length=128),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    access_policy: KnowledgeBaseAccessPolicy = Depends(
        get_agent_access_policy
    ),
    run_control_service: AgentRunControlService = Depends(
        get_agent_run_control_service
    ),
    status_service: AgentThreadStatusService = Depends(
        get_agent_thread_status_service
    ),
) -> AgentThreadStatusResponse:
    """幂等取消 RUNNING / WAITING Stateful Thread。"""

    try:
        _ensure_kb_access(
            db=db,
            user=current_user,
            knowledge_base_id=request.knowledge_base_id,
            access_policy=access_policy,
        )
        run_control_service.cancel(
            db,
            thread_id=thread_id,
            user_id=current_user.id,
            knowledge_base_id=request.knowledge_base_id,
        )
        snapshot = status_service.get_status(
            db,
            thread_id=thread_id,
            user_id=current_user.id,
            knowledge_base_id=request.knowledge_base_id,
        )
        return _build_thread_status_response(snapshot)
    except (ResourceAccessNotFoundError, AgentRunControlNotFoundError) as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except AgentRunStateError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc


@router.post(
    "/threads/{thread_id}/resume",
    response_model=AgentChatResponse | AgentWaitingResponse,
)
def resume_agent_thread(
    request: AgentThreadActionRequest,
    response: Response,
    thread_id: str = ApiPath(min_length=1, max_length=128),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    access_policy: KnowledgeBaseAccessPolicy = Depends(
        get_agent_access_policy
    ),
    runtime_selector: AgentRuntimeSelector = Depends(
        get_agent_runtime_selector
    ),
    status_service: AgentThreadStatusService = Depends(
        get_agent_thread_status_service
    ),
    conversation_service: ConversationService = Depends(
        get_conversation_service
    ),
) -> AgentChatResponse | AgentWaitingResponse:
    """恢复 RUNNING Thread，或恢复已经全部批准的 WAITING Thread。"""

    request_id = get_request_id()
    try:
        runtime_selector.ensure_available(AgentRuntime.LANGGRAPH)
        _ensure_kb_access(
            db=db,
            user=current_user,
            knowledge_base_id=request.knowledge_base_id,
            access_policy=access_policy,
        )
        snapshot = status_service.get_status(
            db,
            thread_id=thread_id,
            user_id=current_user.id,
            knowledge_base_id=request.knowledge_base_id,
        )
        context = _build_thread_authorized_context(
            current_user=current_user,
            knowledge_base_id=request.knowledge_base_id,
            request_id=request_id,
            conversation_id=(
                snapshot.payload.agent_state.conversation.conversation_id
            ),
        )
        runtime_service = cast(
            LangGraphAgentExecutionService,
            runtime_selector.select(AgentRuntime.LANGGRAPH),
        )

        result = runtime_service.resume(
            db=db,
            context=context,
            thread_id=thread_id,
        )
        assistant_message = _append_conversation_message(
            db=db,
            conversation_service=conversation_service,
            user_id=current_user.id,
            conversation_id=context.conversation_id,
            role=ConversationMessageRole.ASSISTANT,
            content=result.answer,
        )
        _extract_completed_turn_memory(
            db=db,
            memory_extraction_service=None,
            user_id=current_user.id,
            knowledge_base_id=request.knowledge_base_id,
            conversation_id=context.conversation_id,
            user_message=None,
            assistant_message=assistant_message,
        )
        return AgentChatResponse(answer=result.answer)

    except AgentInterruptRequired as exc:
        response.status_code = status.HTTP_202_ACCEPTED
        return _build_waiting_response(exc)
    except (
        ResourceAccessNotFoundError,
        AgentThreadNotFoundError,
        AgentResumeCheckpointNotFoundError,
    ) as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except (
        AgentResumeStateError,
        AgentApprovalStateError,
        AgentRunCancellationError,
    ) as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except AgentRuntimeUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent运行时暂不可用",
        ) from exc


@router.post("/threads/{thread_id}/resume/stream")
def stream_resume_agent_thread(
    request: AgentThreadActionRequest,
    thread_id: str = ApiPath(min_length=1, max_length=128),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    access_policy: KnowledgeBaseAccessPolicy = Depends(
        get_agent_access_policy
    ),
    runtime_selector: AgentRuntimeSelector = Depends(
        get_agent_runtime_selector
    ),
    status_service: AgentThreadStatusService = Depends(
        get_agent_thread_status_service
    ),
    conversation_service: ConversationService = Depends(
        get_conversation_service
    ),
) -> StreamingResponse:
    """以既有安全 SSE Contract 继续一个 durable Stateful Thread。"""

    request_id = get_request_id()
    try:
        runtime_selector.ensure_available(AgentRuntime.LANGGRAPH)
        _ensure_kb_access(
            db=db,
            user=current_user,
            knowledge_base_id=request.knowledge_base_id,
            access_policy=access_policy,
        )
        snapshot = status_service.get_status(
            db,
            thread_id=thread_id,
            user_id=current_user.id,
            knowledge_base_id=request.knowledge_base_id,
        )
        context = _build_thread_authorized_context(
            current_user=current_user,
            knowledge_base_id=request.knowledge_base_id,
            request_id=request_id,
            conversation_id=(
                snapshot.payload.agent_state.conversation.conversation_id
            ),
        )
        runtime_service = cast(
            LangGraphAgentExecutionService,
            runtime_selector.select(AgentRuntime.LANGGRAPH),
        )
    except (
        ResourceAccessNotFoundError,
        AgentThreadNotFoundError,
        AgentResumeCheckpointNotFoundError,
    ) as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except (AgentResumeStateError, AgentApprovalStateError) as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except AgentRuntimeUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent运行时暂不可用",
        ) from exc

    return StreamingResponse(
        generate_agent_resume_sse(
            db=db,
            context=context,
            thread_id=thread_id,
            agent_runner=runtime_service,
            conversation_service=conversation_service,
            memory_extraction_service=None,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def generate_agent_resume_sse(
    *,
    db: Session,
    context: ToolExecutionContext,
    thread_id: str,
    agent_runner: LangGraphAgentExecutionService,
    conversation_service: ConversationService | None = None,
    memory_extraction_service: ConversationMemoryExtractionService | None = None,
) -> Iterator[str]:
    """Resume 专用 SSE bridge；只在最终 message 后写用户可见 assistant 历史。"""

    event_stream: Iterator[AgentRunEvent] | None = None
    assistant_answer: str | None = None
    try:
        event_stream = agent_runner.resume_events(
            db=db,
            context=context,
            thread_id=thread_id,
        )
        for event in event_stream:
            if event.type == "message" and event.content:
                assistant_answer = event.content
            yield _encode_agent_sse_event(event)

        if assistant_answer is not None:
            assistant_message = _append_conversation_message(
                db=db,
                conversation_service=conversation_service,
                user_id=context.user_id,
                conversation_id=context.conversation_id,
                role=ConversationMessageRole.ASSISTANT,
                content=assistant_answer,
            )
            _extract_completed_turn_memory(
                db=db,
                memory_extraction_service=None,
                user_id=context.user_id,
                knowledge_base_id=context.knowledge_base_id,
                conversation_id=context.conversation_id,
                user_message=None,
                assistant_message=assistant_message,
            )
        yield "event: done\ndata: {}\n\n"
    except GeneratorExit:
        raise
    except AgentInterruptRequired as exc:
        yield _encode_sse("waiting", _waiting_event_payload(exc))
    except AgentRunCancellationError:
        yield _encode_sse(
            "cancelled",
            {"message": "Agent任务已取消"},
        )
    except (AgentLoopError, LangChainAgentError):
        yield _encode_sse(
            "error",
            {"message": "Agent暂时无法完成请求"},
        )
    except (AgentResumeStateError, AgentApprovalStateError) as exc:
        yield _encode_sse(
            "error",
            {"message": str(exc)},
        )
    except Exception as exc:
        logger.error(
            "Agent resume SSE failed: request_id=%s error_type=%s",
            context.request_id,
            type(exc).__name__,
        )
        yield _encode_sse(
            "error",
            {"message": "Agent恢复执行失败"},
        )
    finally:
        _close_iterator(event_stream)

def generate_agent_chat_sse(
    *,
    db: Session,
    context: ToolExecutionContext,
    message: str,
    agent_runner: AgentRuntimeExecutionService,
    conversation_service: ConversationService | None = None,
    conversation_id: int | None = None,
    memory_extraction_service: ConversationMemoryExtractionService | None = None,
    source_user_message_id: int | None = None,
) -> Iterator[str]:
    """
    把 provider-neutral AgentRunEvent 编码成 SSE。

    正常结束发送 done；流中失败发送 error；客户端取消时向下关闭
    Runtime 事件生成器。不会输出隐藏推理、Tool 参数或 Tool Result 正文。
    """

    event_stream: Iterator[AgentRunEvent] | None = None
    assistant_answer: str | None = None

    try:
        event_stream = agent_runner.run_events(
            db=db,
            context=context,
            message=message,
        )

        for event in event_stream:
            if event.type == "message" and event.content:
                assistant_answer = event.content
            yield _encode_agent_sse_event(event)

        if assistant_answer is not None:
            assistant_message = _append_conversation_message(
                db=db,
                conversation_service=conversation_service,
                user_id=context.user_id,
                conversation_id=conversation_id,
                role=ConversationMessageRole.ASSISTANT,
                content=assistant_answer,
            )
            _extract_completed_turn_memory(
                db=db,
                memory_extraction_service=None,
                user_id=context.user_id,
                knowledge_base_id=context.knowledge_base_id,
                conversation_id=conversation_id,
                user_message_id=source_user_message_id,
                assistant_message=assistant_message,
            )

        yield "event: done\ndata: {}\n\n"

    except GeneratorExit:
        logger.info(
            "Agent SSE cancelled: request_id=%s",
            context.request_id,
        )
        raise

    except AgentInterruptRequired as exc:
        yield _encode_sse(
            "waiting",
            _waiting_event_payload(exc),
        )

    except AgentRunCancellationError:
        yield _encode_sse(
            "cancelled",
            {"message": "Agent任务已取消"},
        )

    except (AgentLoopError, LangChainAgentError) as exc:
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


def _prepare_conversation_persistence(
    *,
    db: Session,
    conversation_service: ConversationService,
    current_user: User,
    conversation_id: int | None,
    knowledge_base_id: int,
    runtime: AgentRuntime,
) -> int | None:
    """绑定可选 Conversation；LangGraph 因 durable Thread 必须显式绑定。"""

    if conversation_id is None:
        if runtime is AgentRuntime.LANGGRAPH:
            raise ValueError("langgraph runtime requires conversation_id")
        return None

    conversation_service.ensure_chat_scope(
        db=db,
        user_id=current_user.id,
        conversation_id=conversation_id,
        mode=ConversationMode.AGENT,
        knowledge_base_id=knowledge_base_id,
    )
    return conversation_id


def _append_conversation_message(
    *,
    db: Session,
    conversation_service: ConversationService | None,
    user_id: int,
    conversation_id: int | None,
    role: ConversationMessageRole,
    content: str,
) -> ConversationMessage | None:
    """只有显式绑定 Conversation 时才写入用户可见历史。"""

    if conversation_service is None or conversation_id is None:
        return None

    return conversation_service.append_message(
        db=db,
        user_id=user_id,
        conversation_id=conversation_id,
        role=role,
        content=content,
    )


def _extract_completed_turn_memory(
    *,
    db: Session,
    memory_extraction_service: ConversationMemoryExtractionService | None,
    user_id: int,
    knowledge_base_id: int,
    conversation_id: int | None,
    assistant_message: ConversationMessage | None,
    user_message: ConversationMessage | None = None,
    user_message_id: int | None = None,
) -> None:
    """成功响应后的 best-effort Memory Extraction，不反向破坏聊天主链路。"""

    if conversation_id is None or assistant_message is None:
        return

    resolved_user_message_id = (
        user_message.id if user_message is not None else user_message_id
    )
    try:
        service = (
            memory_extraction_service
            if memory_extraction_service is not None
            else get_conversation_memory_extraction_service()
        )
        service.extract_completed_turn(
            db,
            user_id=user_id,
            knowledge_base_id=knowledge_base_id,
            conversation_id=conversation_id,
            user_message_id=resolved_user_message_id,
            assistant_message_id=assistant_message.id,
        )
    except ConversationMemoryExtractionError as exc:
        logger.warning(
            "Conversation memory extraction skipped: conversation_id=%s "
            "error_type=%s",
            conversation_id,
            type(exc).__name__,
        )
    except Exception as exc:
        logger.error(
            "Conversation memory persistence failed: conversation_id=%s "
            "error_type=%s",
            conversation_id,
            type(exc).__name__,
        )


def _build_authorized_context(
    *,
    db: Session,
    request: AgentChatRequest,
    current_user: User,
    access_policy: KnowledgeBaseAccessPolicy,
    request_id: str,
    conversation_id: int | None = None,
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
        conversation_id=conversation_id,
    )


def _ensure_kb_access(
    *,
    db: Session,
    user: User,
    knowledge_base_id: int,
    access_policy: KnowledgeBaseAccessPolicy,
) -> None:
    access_policy.get_accessible_knowledge_base(
        db=db,
        knowledge_base_id=knowledge_base_id,
        user=user,
    )


def _build_thread_authorized_context(
    *,
    current_user: User,
    knowledge_base_id: int,
    request_id: str,
    conversation_id: int,
) -> ToolExecutionContext:
    return ToolExecutionContext(
        user_id=current_user.id,
        role=UserRole(current_user.role),
        knowledge_base_id=knowledge_base_id,
        request_id=request_id,
        conversation_id=conversation_id,
    )


def _build_waiting_response(
    exc: AgentInterruptRequired,
) -> AgentWaitingResponse:
    return AgentWaitingResponse(
        thread_id=exc.thread_id,
        approvals=[
            AgentApprovalRequirementResponse(
                call_id=item.call_id,
                tool_name=item.tool_name,
                reason=item.reason,
            )
            for item in exc.approvals
        ],
    )


def _waiting_event_payload(
    exc: AgentInterruptRequired,
) -> dict[str, Any]:
    response = _build_waiting_response(exc)
    return response.model_dump(mode="json")


def _build_thread_status_response(
    snapshot: AgentThreadStatusSnapshot,
) -> AgentThreadStatusResponse:
    payload = snapshot.payload
    state = payload.agent_state
    return AgentThreadStatusResponse(
        thread_id=state.thread.thread_id,
        conversation_id=state.conversation.conversation_id,
        knowledge_base_id=state.conversation.knowledge_base_id,
        status=state.status,
        retry_count=state.retry_count,
        last_error_code=state.last_error_code,
        pending_approvals=[
            AgentApprovalRequirementResponse(
                call_id=item.call_id,
                tool_name=item.tool_name,
                reason=item.reason,
            )
            for item in payload.pending_approvals
        ],
        can_resume=snapshot.can_resume,
        can_approve=snapshot.can_approve,
        can_reject=snapshot.can_reject,
        can_cancel=snapshot.can_cancel,
    )


def _encode_agent_sse_event(event: AgentRunEvent) -> str:
    """把 Runtime 事件转换成对外最小、安全的 SSE Payload。"""

    if event.type == "message":
        payload: dict[str, Any] = {
            "content": event.content,
        }
    else:
        payload = event.model_dump(
            exclude={"type", "duration_ms"},
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
