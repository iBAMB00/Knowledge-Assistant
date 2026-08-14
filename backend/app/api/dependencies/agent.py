from functools import lru_cache

from app.agent.native_agent import NativeAgentRunner
from app.agent.tools.knowledge_search import KnowledgeSearchTool
from app.repositories.document_repository import DocumentRepository
from app.repositories.knowledge_base_repository import KnowledgeBaseRepository
from app.services.knowledge_base_access_policy import KnowledgeBaseAccessPolicy
from app.services.retrieval_service import RetrievalService


@lru_cache
def get_agent_access_policy() -> KnowledgeBaseAccessPolicy:
    """获取 Agent HTTP 入口与 Tool 共享的知识库访问策略。"""

    return KnowledgeBaseAccessPolicy(
        knowledge_base_repository=KnowledgeBaseRepository(),
        document_repository=DocumentRepository(),
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


@lru_cache
def get_native_agent_runner() -> NativeAgentRunner:
    """构建当前 v2.0 Native Agent Runner。"""

    from app.services.llm_service import LLMService

    search_knowledge_tool = KnowledgeSearchTool(
        retrieval_service=get_agent_retrieval_service(),
        access_policy=get_agent_access_policy(),
    )

    return NativeAgentRunner(
        llm_service=LLMService(),
        tools=[search_knowledge_tool],
    )
