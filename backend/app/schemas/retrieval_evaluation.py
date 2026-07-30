from typing import Literal

from pydantic import BaseModel, Field, PositiveInt


RetrievalEvaluationMode = Literal[
    "baseline",
    "optimized",
]


class RetrievalEvaluationCase(BaseModel):
    """
    单条检索评估问题。
    """

    case_id: str

    question: str

    expected_document_ids: list[
        PositiveInt
    ] = Field(
        min_length=1,
    )

    document_id: PositiveInt | None = None


class RetrievalEvaluationCaseResult(BaseModel):
    """
    单条问题的检索评估结果。
    """

    case_id: str

    question: str

    retrieval_mode: RetrievalEvaluationMode

    expected_document_ids: list[int]

    retrieved_document_ids: list[int]

    retrieved_chunk_ids: list[int]

    hit: bool

    reciprocal_rank: float

    document_coverage: float

    duplicate_rate: float

    latency_ms: float


class RetrievalEvaluationSummary(BaseModel):
    """
    单种检索模式的汇总指标。
    """

    retrieval_mode: RetrievalEvaluationMode

    total_cases: int

    hit_rate_at_k: float

    mean_reciprocal_rank: float

    mean_document_coverage: float

    full_document_coverage_rate_at_k: float

    mean_duplicate_rate: float

    average_latency_ms: float


class RetrievalEvaluationRun(BaseModel):
    """
    单种检索模式的完整评估结果。
    """

    summary: RetrievalEvaluationSummary

    cases: list[RetrievalEvaluationCaseResult]


class RetrievalComparisonReport(BaseModel):
    """
    Baseline与Optimized对比报告。
    """

    baseline: RetrievalEvaluationRun

    optimized: RetrievalEvaluationRun