from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.retrieval_evaluation import (
    RetrievalEvaluationDatasetReference,
    RetrievalEvaluationRun,
    RetrievalEvaluationMode,
)


class RetrievalAblationComponent(str, Enum):
    """当前生产链路可被单独关闭的检索组件。"""

    HYBRID_RRF = "hybrid_rrf"
    RERANKER = "reranker"
    PARENT_CHILD = "parent_child"


class RetrievalAblationEvidenceStrength(str, Enum):
    """消融结论的因果解释强度。"""

    CONTROLLED = "controlled"
    RUNTIME_ONLY = "runtime_only"
    REFERENCE = "reference"


class RetrievalAblationVariant(BaseModel):
    """一个可审计的 Retrieval 实验变体。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    variant_id: str = Field(min_length=1, max_length=100)
    label: str = Field(min_length=1, max_length=200)
    retrieval_mode: RetrievalEvaluationMode
    parent_child_enabled: bool
    hybrid_rrf_enabled: bool
    reranker_enabled: bool
    removed_component: RetrievalAblationComponent | None = None
    evidence_strength: RetrievalAblationEvidenceStrength
    notes: str | None = Field(default=None, max_length=1000)


class RetrievalAblationConfiguration(BaseModel):
    """所有变体必须共享的控制变量。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code_version: str | None = Field(default=None, max_length=100)
    vector_store_backend: Literal[
        "database",
        "qdrant",
    ]
    embedding_provider: str = Field(min_length=1, max_length=100)
    embedding_model: str = Field(min_length=1, max_length=200)
    top_k: int = Field(gt=0)
    candidate_k: int = Field(gt=0)
    score_threshold: float = Field(ge=-1.0, le=1.0)
    per_document_limit: int = Field(gt=0)
    shared_query_embedding: bool = True
    reranker_model: str | None = Field(default=None, max_length=200)
    reranker_fail_open: bool = True
    cost_currency: str = Field(default="CNY", min_length=1, max_length=16)
    reranker_price_per_million_tokens: float = Field(default=0.0, ge=0.0)


class RetrievalAblationSharedTokenUsage(BaseModel):
    """整个 Ablation 共享的 Query Embedding Token 统计。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    token_count_source: Literal["local_estimation"] = "local_estimation"
    tokenizer_name: str = Field(min_length=1, max_length=100)
    request_count: int = Field(ge=0)
    total_query_embedding_tokens: int = Field(ge=0)
    average_query_embedding_tokens: float = Field(ge=0.0)
    p95_query_embedding_tokens: float = Field(ge=0.0)


class RetrievalAblationRerankerTokenUsage(BaseModel):
    """单个 Variant 的 Reranker Provider Token 统计。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    request_count: int = Field(ge=0)
    successful_request_count: int = Field(ge=0)
    failed_request_count: int = Field(ge=0)
    candidate_count: int = Field(ge=0)
    average_candidates_per_request: float = Field(ge=0.0)
    provider_usage_request_count: int = Field(ge=0)
    provider_total_tokens: int | None = Field(default=None, ge=0)
    average_provider_tokens_per_reported_request: float | None = Field(
        default=None, ge=0.0
    )
    p95_provider_tokens_per_reported_request: float | None = Field(
        default=None, ge=0.0
    )
    usage_complete: bool
    token_count_source: Literal[
        "provider_usage",
        "partial_provider_usage",
        "unavailable",
        "not_applicable",
    ]
    reported_provider_token_cost: float | None = Field(
        default=None, ge=0.0
    )


class RetrievalAblationContextTokenUsage(BaseModel):
    """单个 Variant 最终 Top-K Context 的本地 Token 估算。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    token_count_source: Literal["local_estimation"] = "local_estimation"
    tokenizer_name: str = Field(min_length=1, max_length=100)
    total_context_tokens: int = Field(ge=0)
    average_context_tokens: float = Field(ge=0.0)
    p50_context_tokens: float = Field(ge=0.0)
    p95_context_tokens: float = Field(ge=0.0)


class RetrievalAblationVariantTokenUsage(BaseModel):
    """一个 Variant 的 Reranker 与最终 Context 分阶段 Token。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    reranker: RetrievalAblationRerankerTokenUsage
    final_context: RetrievalAblationContextTokenUsage | None = None


class RetrievalAblationMetricSnapshot(BaseModel):
    """简历/报告最常用的质量、成本和延迟快照。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_mrr: float = Field(ge=0.0, le=1.0)
    document_recall_at_k: float = Field(ge=0.0, le=1.0)
    chunk_hit_rate_at_k: float = Field(ge=0.0, le=1.0)
    chunk_mrr: float = Field(ge=0.0, le=1.0)
    chunk_recall_at_k: float = Field(ge=0.0, le=1.0)
    chunk_ndcg_at_k: float = Field(ge=0.0, le=1.0)
    no_answer_accuracy: float = Field(ge=0.0, le=1.0)
    average_context_tokens: float | None = Field(default=None, ge=0.0)
    average_retrieval_latency_ms: float = Field(ge=0.0)
    p95_total_latency_ms: float = Field(ge=0.0)


class RetrievalAblationVariantResult(BaseModel):
    """一个变体的完整 Retrieval Evaluation 结果。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    variant: RetrievalAblationVariant
    metrics: RetrievalAblationMetricSnapshot
    token_usage: RetrievalAblationVariantTokenUsage
    run: RetrievalEvaluationRun


class RetrievalAblationDelta(BaseModel):
    """Full 与某个移除组件变体之间的指标差异。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    variant_id: str = Field(min_length=1, max_length=100)
    removed_component: RetrievalAblationComponent | None
    evidence_strength: RetrievalAblationEvidenceStrength
    metric_id: str = Field(min_length=1, max_length=100)
    full_value: float
    variant_value: float
    raw_delta_full_minus_variant: float
    full_advantage: float
    higher_is_better: bool


class RetrievalAblationCaseRegression(BaseModel):
    """移除组件后出现的逐 Case 质量退化。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    variant_id: str = Field(min_length=1, max_length=100)
    case_id: str = Field(min_length=1, max_length=100)
    removed_component: RetrievalAblationComponent | None
    evidence_strength: RetrievalAblationEvidenceStrength
    reasons: tuple[str, ...] = Field(min_length=1)
    full_chunk_hit: bool | None = None
    variant_chunk_hit: bool | None = None
    full_chunk_recall_at_k: float | None = Field(default=None, ge=0.0, le=1.0)
    variant_chunk_recall_at_k: float | None = Field(default=None, ge=0.0, le=1.0)
    full_chunk_ndcg_at_k: float | None = Field(default=None, ge=0.0, le=1.0)
    variant_chunk_ndcg_at_k: float | None = Field(default=None, ge=0.0, le=1.0)



class RetrievalAblationReport(BaseModel):
    """A7 可持久化 Retrieval 消融实验报告。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    generated_at: datetime
    dataset: RetrievalEvaluationDatasetReference
    configuration: RetrievalAblationConfiguration
    full_variant_id: str = Field(min_length=1, max_length=100)
    shared_token_usage: RetrievalAblationSharedTokenUsage
    variants: tuple[RetrievalAblationVariantResult, ...] = Field(min_length=2)
    deltas: tuple[RetrievalAblationDelta, ...] = ()
    case_regressions: tuple[RetrievalAblationCaseRegression, ...] = ()
    limitations: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_report(self) -> "RetrievalAblationReport":
        variant_ids = [result.variant.variant_id for result in self.variants]
        if len(set(variant_ids)) != len(variant_ids):
            raise ValueError("ablation variant_id must be unique")
        if self.full_variant_id not in variant_ids:
            raise ValueError("full_variant_id must reference a variant")
        return self
