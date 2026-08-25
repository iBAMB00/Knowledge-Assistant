from functools import lru_cache

from app.agent.agent_prompt import AGENT_TOOL_CALLING_SYSTEM_PROMPT
from app.agent.frameworks.langchain.runner import LangChainSingleAgentRunner
from app.agent.frameworks.langgraph.runner import LangGraphStatefulRunner
from app.agent.native_agent import NativeAgentRunner
from app.agent.observability.factory import build_observability_provider
from app.agent.observability.provider import ObservabilityProvider
from app.agent.version_snapshot import build_agent_runtime_version_snapshot
from app.agent.tools.document_get import DocumentGetTool
from app.agent.tools.document_list import DocumentListTool
from app.agent.tools.knowledge_base_list import KnowledgeBaseListTool
from app.agent.tools.knowledge_search import KnowledgeSearchTool
from app.agent.tools.processing_job_get import ProcessingJobGetTool
from app.agent.tools.base import BaseAgentTool
from app.api.dependencies.mcp import get_mcp_runtime_tools
from app.api.dependencies.conversation import get_conversation_service
from app.repositories.agent_run_repository import AgentRunRepository
from app.repositories.agent_tool_call_repository import AgentToolCallRepository
from app.repositories.document_chunk_repository import DocumentChunkRepository
from app.repositories.document_content_repository import DocumentContentRepository
from app.repositories.document_repository import DocumentRepository
from app.repositories.knowledge_base_repository import KnowledgeBaseRepository
from app.repositories.processing_job_repository import ProcessingJobRepository
from app.repositories.conversation_summary_repository import (
    ConversationSummaryRepository,
)
from app.repositories.conversation_message_repository import (
    ConversationMessageRepository,
)
from app.repositories.memory_item_repository import MemoryItemRepository
from app.services.agent_execution_service import AgentExecutionService
from app.services.agent_checkpoint_service import AgentCheckpointService
from app.services.agent_hitl_service import AgentHITLService
from app.services.agent_recovery_service import AgentRecoveryService
from app.services.agent_run_control_service import AgentRunControlService
from app.services.agent_thread_status_service import AgentThreadStatusService
from app.services.agent_run_query_service import AgentRunQueryService
from app.services.document_operation_policy import DocumentOperationPolicy
from app.services.document_service import DocumentService
from app.services.knowledge_base_access_policy import KnowledgeBaseAccessPolicy
from app.services.langchain_agent_execution_service import (
    LangChainAgentExecutionService,
)
from app.services.langgraph_agent_execution_service import (
    LangGraphAgentExecutionService,
)
from app.services.agent_runtime_diagnostics_service import (
    AgentRuntimeDiagnosticsService,
)
from app.services.agent_runtime_selector import AgentRuntimeSelector
from app.services.conversation_history_context_provider import (
    ConversationHistoryContextProvider,
)
from app.services.conversation_context_provider import ConversationContextProvider
from app.services.conversation_summary_service import (
    ConversationSummaryService,
    LLMConversationSummaryGenerator,
)
from app.services.conversation_memory_service import ConversationMemoryService
from app.services.conversation_memory_context_provider import (
    ConversationMemoryContextProvider,
    LexicalConversationMemoryRetriever,
)
from app.services.conversation_memory_extraction_service import (
    ConversationMemoryExtractionService,
    LLMConversationMemoryExtractor,
)
from app.services.knowledge_base_service import KnowledgeBaseService
from app.services.processing_job_service import ProcessingJobService
from app.services.retrieval_service import RetrievalService


@lru_cache
def get_agent_observability_provider() -> ObservabilityProvider:
    """Build the app-wide fail-open Agent observability provider."""

    from app.core.config import get_settings

    return build_observability_provider(get_settings())


@lru_cache
def get_agent_access_policy() -> KnowledgeBaseAccessPolicy:
    """获取 Agent HTTP 入口与 Tool 共享的知识库访问策略。"""

    return KnowledgeBaseAccessPolicy(
        knowledge_base_repository=KnowledgeBaseRepository(),
        document_repository=DocumentRepository(),
    )


@lru_cache
def get_agent_knowledge_base_service() -> KnowledgeBaseService:
    """构建 list_knowledge_bases Tool 复用的 KnowledgeBaseService。"""

    knowledge_base_repository = KnowledgeBaseRepository()
    document_repository = DocumentRepository()
    access_policy = KnowledgeBaseAccessPolicy(
        knowledge_base_repository=knowledge_base_repository,
        document_repository=document_repository,
    )
    return KnowledgeBaseService(
        knowledge_base_repository=knowledge_base_repository,
        document_repository=document_repository,
        access_policy=access_policy,
    )


@lru_cache
def get_agent_document_service() -> DocumentService:
    """构建文档只读 Tool 复用的现有 DocumentService。"""

    from app.services.storage.factory import get_storage_service

    document_repository = DocumentRepository()
    processing_job_repository = ProcessingJobRepository()

    return DocumentService(
        storage_service=get_storage_service(),
        document_repository=document_repository,
        document_content_repository=DocumentContentRepository(),
        document_chunk_repository=DocumentChunkRepository(),
        processing_job_repository=processing_job_repository,
        document_operation_policy=DocumentOperationPolicy(
            processing_job_repository=processing_job_repository,
        ),
        vector_index=None,
    )


@lru_cache
def get_agent_processing_job_service() -> ProcessingJobService:
    """构建 get_processing_job Tool 复用的 ProcessingJobService。"""

    return ProcessingJobService(
        document_repository=DocumentRepository(),
        processing_job_repository=ProcessingJobRepository(),
    )


@lru_cache
def get_agent_retrieval_service() -> RetrievalService:
    """
    构建 Agent search_knowledge Tool 复用的 RetrievalService。

    重依赖延迟到函数调用时加载，避免仅导入 Agent Router
    就提前初始化模型、Qdrant 等运行时组件。
    """

    from app.core.config import get_settings
    from app.repositories.document_chunk_repository import DocumentChunkRepository
    from app.services.bm25_retrieval_service import BM25RetrievalService
    from app.services.embedding.factory import EmbeddingFactory
    from app.services.reranker.factory import RerankerFactory
    from app.services.rrf_fusion_service import RRFFusionService
    from app.services.vector_store.factory import get_vector_store_components

    settings = get_settings()
    document_chunk_repository = DocumentChunkRepository()

    reranker = (
        RerankerFactory.create()
        if settings.reranker_enabled
        else None
    )

    return RetrievalService(
        embedding_provider=EmbeddingFactory.create(),
        vector_store=get_vector_store_components().vector_store,
        default_top_k=settings.retrieval_top_k,
        default_candidate_k=settings.retrieval_candidate_k,
        default_score_threshold=settings.knowledge_chat_score_threshold,
        default_per_document_limit=settings.retrieval_per_document_limit,
        document_chunk_repository=document_chunk_repository,
        parent_child_enabled=settings.parent_child_enabled,
        bm25_retriever=BM25RetrievalService(
            document_chunk_repository=document_chunk_repository,
        ),
        rrf_fusion_service=RRFFusionService(
            rank_constant=settings.retrieval_rrf_k,
        ),
        hybrid_enabled=settings.retrieval_hybrid_enabled,
        reranker=reranker,
        reranker_enabled=settings.reranker_enabled,
        reranker_fail_open=settings.reranker_fail_open,
    )


def _merge_agent_tools(
    local_tools: tuple[BaseAgentTool, ...],
    mcp_tools: tuple[BaseAgentTool, ...],
) -> tuple[BaseAgentTool, ...]:
    """Merge Local/MCP Tools while rejecting model-visible name collisions."""

    tools = local_tools + mcp_tools
    names = [tool.name for tool in tools]
    if len(names) != len(set(names)):
        raise RuntimeError("duplicate Agent runtime tool name")
    return tools


def get_agent_tools() -> tuple[BaseAgentTool, ...]:
    """构建 Native / Framework Candidate 共用的 Local + MCP Tool 集合。"""

    access_policy = get_agent_access_policy()
    document_service = get_agent_document_service()

    local_tools = (
        KnowledgeSearchTool(
            retrieval_service=get_agent_retrieval_service(),
            access_policy=access_policy,
        ),
        KnowledgeBaseListTool(
            knowledge_base_service=get_agent_knowledge_base_service(),
        ),
        DocumentListTool(
            document_service=document_service,
            access_policy=access_policy,
        ),
        DocumentGetTool(
            document_service=document_service,
            access_policy=access_policy,
        ),
        ProcessingJobGetTool(
            processing_job_service=get_agent_processing_job_service(),
            access_policy=access_policy,
        ),
    )

    return _merge_agent_tools(
        local_tools=local_tools,
        mcp_tools=tuple(get_mcp_runtime_tools()),
    )


@lru_cache
def get_conversation_history_context_provider(
) -> ConversationHistoryContextProvider:
    """构建三条 Agent Runtime 共用的 Conversation History Provider。"""

    return ConversationHistoryContextProvider(
        conversation_service=get_conversation_service(),
    )


@lru_cache
def get_conversation_summary_service() -> ConversationSummaryService:
    """构建 B5 Conversation Summary 增量生成服务。"""

    from app.services.llm_service import LLMService

    return ConversationSummaryService(
        repository=ConversationSummaryRepository(),
        generator=LLMConversationSummaryGenerator(
            llm_service=LLMService(),
        ),
    )


@lru_cache
def get_conversation_context_provider() -> ConversationContextProvider:
    """组合 Summary + Summary 边界后的 Raw History。"""

    return ConversationContextProvider(
        history_provider=get_conversation_history_context_provider(),
        summary_service=get_conversation_summary_service(),
        memory_context_provider=get_conversation_memory_context_provider(),
    )


@lru_cache
def get_conversation_memory_service() -> ConversationMemoryService:
    """构建 B6 Conversation-scoped Memory 持久化服务。"""

    return ConversationMemoryService(
        repository=MemoryItemRepository(),
        message_repository=ConversationMessageRepository(),
        conversation_service=get_conversation_service(),
    )


@lru_cache
def get_conversation_memory_context_provider(
) -> ConversationMemoryContextProvider:
    """构建 B8 Conversation-scoped Memory 检索与 Context Provider。"""

    return ConversationMemoryContextProvider(
        memory_service=get_conversation_memory_service(),
        retriever=LexicalConversationMemoryRetriever(),
    )


@lru_cache
def get_conversation_memory_extraction_service(
) -> ConversationMemoryExtractionService:
    """构建 B7 Memory Extraction 生产服务。"""

    from app.services.llm_service import LLMService

    return ConversationMemoryExtractionService(
        extractor=LLMConversationMemoryExtractor(
            llm_service=LLMService(),
        ),
        memory_service=get_conversation_memory_service(),
        message_repository=ConversationMessageRepository(),
    )


@lru_cache
def get_native_agent_runner() -> NativeAgentRunner:
    """构建当前 v2.0 Native Agent Runner。"""

    from app.services.llm_service import LLMService

    return NativeAgentRunner(
        llm_service=LLMService(),
        tools=get_agent_tools(),
    )


@lru_cache
def get_agent_execution_service() -> AgentExecutionService:
    """构建带 AgentRun / ToolCall 持久化的执行服务。"""

    from app.core.config import get_settings

    from app.services.llm_service import LLMService

    settings = get_settings()
    agent_runner = get_native_agent_runner()
    version_snapshot = build_agent_runtime_version_snapshot(
        settings=settings,
        tool_contracts=agent_runner.tool_contracts,
        prompt_version=LLMService.AGENT_PROMPT_VERSION,
    )
    return AgentExecutionService(
        agent_runner=agent_runner,
        agent_run_repository=AgentRunRepository(),
        tool_call_repository=AgentToolCallRepository(),
        model_provider=settings.model_provider,
        model_name=settings.model_name,
        version_snapshot=version_snapshot,
        conversation_history_provider=(
            get_conversation_context_provider()
        ),
        observability_provider=get_agent_observability_provider(),
    )


@lru_cache
def get_langchain_agent_runner() -> LangChainSingleAgentRunner:
    """构建 LangChain Candidate Runner；仅供显式 Runtime 选择与评估使用。"""

    from app.agent.frameworks.langchain.model_adapter import LangChainModelAdapter
    from app.core.config import get_settings

    settings = get_settings()
    return LangChainSingleAgentRunner(
        model=LangChainModelAdapter(settings).build(),
        tools=get_agent_tools(),
    )


@lru_cache
def get_langchain_agent_execution_service() -> LangChainAgentExecutionService:
    """构建带 AgentRun / ToolCall 持久化的 LangChain Candidate 执行服务。"""

    from app.core.config import get_settings

    settings = get_settings()
    agent_runner = get_langchain_agent_runner()
    version_snapshot = build_agent_runtime_version_snapshot(
        settings=settings,
        tool_contracts=agent_runner.tool_contracts,
        prompt_version=AGENT_TOOL_CALLING_SYSTEM_PROMPT.version,
        agent_version=f"langchain-v1:{agent_runner.RUNNER_VERSION}",
    )
    return LangChainAgentExecutionService(
        agent_runner=agent_runner,
        agent_run_repository=AgentRunRepository(),
        tool_call_repository=AgentToolCallRepository(),
        model_provider=settings.model_provider,
        model_name=settings.model_name,
        version_snapshot=version_snapshot,
        conversation_history_provider=(
            get_conversation_context_provider()
        ),
        observability_provider=get_agent_observability_provider(),
    )


@lru_cache
def get_agent_checkpoint_service() -> AgentCheckpointService:
    """构建 Stateful Thread / Checkpoint 事务服务。"""

    return AgentCheckpointService()


@lru_cache
def get_agent_recovery_service() -> AgentRecoveryService:
    return AgentRecoveryService(
        checkpoint_service=get_agent_checkpoint_service(),
    )


@lru_cache
def get_agent_hitl_service() -> AgentHITLService:
    return AgentHITLService(
        checkpoint_service=get_agent_checkpoint_service(),
    )


@lru_cache
def get_agent_run_control_service() -> AgentRunControlService:
    return AgentRunControlService(
        checkpoint_service=get_agent_checkpoint_service(),
    )


@lru_cache
def get_agent_thread_status_service() -> AgentThreadStatusService:
    return AgentThreadStatusService(
        checkpoint_service=get_agent_checkpoint_service(),
    )


@lru_cache
def get_langgraph_agent_runner() -> LangGraphStatefulRunner:
    """构建 v2.3 Stateful Candidate，继续共享现有 Local + MCP Toolset。"""

    from app.services.llm_service import LLMService

    return LangGraphStatefulRunner(
        llm_service=LLMService(),
        tools=get_agent_tools(),
        checkpoint_writer=get_agent_checkpoint_service(),
        recovery_loader=get_agent_recovery_service(),
        hitl_loader=get_agent_hitl_service(),
        cancellation_probe=get_agent_run_control_service(),
    )


@lru_cache
def get_langgraph_agent_execution_service() -> LangGraphAgentExecutionService:
    """把 Stateful Runner 接入 AgentRun / ToolCall production lifecycle。"""

    from app.core.config import get_settings
    from app.services.llm_service import LLMService

    settings = get_settings()
    agent_runner = get_langgraph_agent_runner()
    version_snapshot = build_agent_runtime_version_snapshot(
        settings=settings,
        tool_contracts=agent_runner.tool_contracts,
        prompt_version=AGENT_TOOL_CALLING_SYSTEM_PROMPT.version,
        agent_version=f"langgraph-v1:{agent_runner.RUNNER_VERSION}",
    )
    return LangGraphAgentExecutionService(
        agent_runner=agent_runner,
        agent_run_repository=AgentRunRepository(),
        tool_call_repository=AgentToolCallRepository(),
        model_provider=settings.model_provider,
        model_name=settings.model_name,
        version_snapshot=version_snapshot,
        checkpoint_service=get_agent_checkpoint_service(),
        recovery_service=get_agent_recovery_service(),
        hitl_service=get_agent_hitl_service(),
        conversation_history_provider=(
            get_conversation_context_provider()
        ),
        observability_provider=get_agent_observability_provider(),
    )


def reset_agent_runtime_caches() -> None:
    """Drop Runner/Service snapshots after MCP startup or shutdown changes Toolset."""

    get_conversation_memory_service.cache_clear()
    get_conversation_memory_context_provider.cache_clear()
    get_conversation_context_provider.cache_clear()
    get_conversation_memory_extraction_service.cache_clear()
    get_native_agent_runner.cache_clear()
    get_agent_execution_service.cache_clear()
    get_langchain_agent_runner.cache_clear()
    get_langchain_agent_execution_service.cache_clear()
    get_langgraph_agent_runner.cache_clear()
    get_langgraph_agent_execution_service.cache_clear()
    get_agent_runtime_selector.cache_clear()


@lru_cache
def get_agent_runtime_selector() -> AgentRuntimeSelector:
    """构建 HTTP 同步入口的 Runtime 选择器；默认 Native，Candidate 显式开放。"""

    from app.core.config import get_settings

    settings = get_settings()
    return AgentRuntimeSelector(
        native_factory=get_agent_execution_service,
        langchain_factory=get_langchain_agent_execution_service,
        langchain_candidate_enabled=(
            settings.agent_langchain_candidate_enabled
        ),
        langgraph_factory=get_langgraph_agent_execution_service,
        langgraph_candidate_enabled=(
            settings.agent_langgraph_candidate_enabled
        ),
    )


@lru_cache
def get_agent_runtime_diagnostics_service() -> AgentRuntimeDiagnosticsService:
    """构建不初始化模型/Tool 的 Agent Runtime 诊断服务。"""

    from app.core.config import get_settings

    settings = get_settings()
    return AgentRuntimeDiagnosticsService(
        langchain_candidate_enabled=(
            settings.agent_langchain_candidate_enabled
        ),
        langgraph_candidate_enabled=(
            settings.agent_langgraph_candidate_enabled
        ),
    )


@lru_cache
def get_agent_run_query_service() -> AgentRunQueryService:
    """构建 AgentRun 只读查询服务。"""

    return AgentRunQueryService(
        agent_run_repository=AgentRunRepository(),
        tool_call_repository=AgentToolCallRepository(),
    )
