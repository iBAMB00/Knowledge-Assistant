import hashlib
import json
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.agent.tools.base import ToolContract
from app.core.config import Settings


AGENT_RUNTIME_VERSION = "2.0.0"
_VERSION_FINGERPRINT_LENGTH = 16


class AgentRuntimeVersionSnapshot(BaseModel):
    """一次 Agent Run 的稳定运行时版本快照。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    agent_version: str = Field(min_length=1, max_length=64)
    prompt_version: str = Field(min_length=1, max_length=64)
    toolset_version: str = Field(min_length=1, max_length=64)
    retrieval_config_version: str = Field(min_length=1, max_length=64)


class AgentEvaluationVersionContext(BaseModel):
    """仅 Eval Run 使用的评估版本上下文。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_version: str = Field(min_length=1, max_length=50)
    evaluator_version: str = Field(min_length=1, max_length=50)


def build_agent_runtime_version_snapshot(
    *,
    settings: Settings,
    tool_contracts: Sequence[ToolContract],
    prompt_version: str,
    agent_version: str = AGENT_RUNTIME_VERSION,
) -> AgentRuntimeVersionSnapshot:
    """根据当前 Tool Contract 与 Retrieval 配置构建运行时版本快照。"""

    normalized_agent_version = agent_version.strip()
    if not normalized_agent_version:
        raise ValueError("agent_version cannot be empty")

    return AgentRuntimeVersionSnapshot(
        agent_version=normalized_agent_version,
        prompt_version=prompt_version.strip(),
        toolset_version=build_toolset_version(tool_contracts),
        retrieval_config_version=build_retrieval_config_version(settings),
    )


def build_toolset_version(
    tool_contracts: Sequence[ToolContract],
) -> str:
    """对当前 Tool Contract 集合生成顺序无关的稳定指纹。"""

    normalized = [
        contract.model_dump(mode="json")
        for contract in sorted(tool_contracts, key=lambda item: item.name)
    ]
    return _fingerprint("toolset-v2", normalized)


def build_retrieval_config_version(settings: Settings) -> str:
    """仅对会影响当前 Agent 检索行为的非敏感配置生成稳定指纹。"""

    payload: dict[str, Any] = {
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.embedding_model,
        "embedding_dimension": settings.embedding_dimension,
        "vector_store_backend": settings.vector_store_backend,
        "qdrant_collection_name": settings.qdrant_collection_name,
        "retrieval_top_k": settings.retrieval_top_k,
        "retrieval_candidate_k": settings.retrieval_candidate_k,
        "knowledge_chat_score_threshold": settings.knowledge_chat_score_threshold,
        "retrieval_per_document_limit": settings.retrieval_per_document_limit,
        "retrieval_hybrid_enabled": settings.retrieval_hybrid_enabled,
        "retrieval_rrf_k": settings.retrieval_rrf_k,
        "parent_child_enabled": settings.parent_child_enabled,
        "reranker_enabled": settings.reranker_enabled,
        "reranker_provider": settings.reranker_provider,
        "reranker_model": settings.reranker_model,
        "reranker_fail_open": settings.reranker_fail_open,
        "reranker_instruct": settings.reranker_instruct,
    }
    return _fingerprint("retrieval-v1", payload)


def _fingerprint(prefix: str, payload: Any) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()[:_VERSION_FINGERPRINT_LENGTH]
    return f"{prefix}:{digest}"
