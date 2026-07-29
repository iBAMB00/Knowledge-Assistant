from sqlalchemy.orm import Session
import pytest

from app.schemas.retrieval_evaluation import (
    RetrievalEvaluationCase,
)
from app.schemas.vector_search_result import (
    VectorSearchResult,
)
from app.services.evaluation.retrieval_evaluator import (
    RetrievalEvaluator,
)


class FakeRetrievalService:
    """
    检索评估测试使用的服务。
    """

    def __init__(self) -> None:
        self.received_modes: list[str] = []

    def retrieve(
        self,
        db: Session,
        query: str,
        top_k: int | None = None,
        candidate_k: int | None = None,
        score_threshold: float | None = None,
        per_document_limit: int | None = None,
        document_id: int | None = None,
        retrieval_mode: str = "optimized",
    ) -> list[VectorSearchResult]:
        """
        根据检索模式返回不同结果。
        """

        self.received_modes.append(
            retrieval_mode
        )

        if query == "没有匹配的问题":
            return []

        if retrieval_mode == "baseline":
            return [
                VectorSearchResult(
                    document_id=1,
                    chunk_id=1,
                    chunk_index=0,
                    content="重复内容",
                    score=0.99,
                ),
                VectorSearchResult(
                    document_id=1,
                    chunk_id=2,
                    chunk_index=1,
                    content=" 重复内容 ",
                    score=0.98,
                ),
                VectorSearchResult(
                    document_id=2,
                    chunk_id=3,
                    chunk_index=0,
                    content="预期文档二",
                    score=0.90,
                ),
            ]

        return [
            VectorSearchResult(
                document_id=2,
                chunk_id=3,
                chunk_index=0,
                content="预期文档二",
                score=0.90,
            ),
            VectorSearchResult(
                document_id=3,
                chunk_id=4,
                chunk_index=0,
                content="预期文档三",
                score=0.88,
            ),
        ]


def build_evaluation_case(
    case_id: str = "multi-document-001",
    question: str = "多文档测试问题",
) -> RetrievalEvaluationCase:
    """
    创建测试评估问题。
    """

    return RetrievalEvaluationCase(
        case_id=case_id,
        question=question,
        expected_document_ids=[
            2,
            3,
        ],
    )


def test_evaluate_calculates_baseline_metrics(
    db: Session,
) -> None:
    """
    验证Baseline指标计算。
    """

    evaluator = RetrievalEvaluator(
        retrieval_service=FakeRetrievalService(),
    )

    evaluation_run = evaluator.evaluate(
        db=db,
        cases=[
            build_evaluation_case(),
        ],
        retrieval_mode="baseline",
        top_k=3,
    )

    case_result = evaluation_run.cases[0]

    assert case_result.hit is True

    assert case_result.reciprocal_rank == (
        1.0 / 3
    )

    assert case_result.document_coverage == 0.5

    assert case_result.duplicate_rate == (
        1.0 / 3
    )

    assert case_result.latency_ms >= 0.0

    assert (
        evaluation_run.summary.hit_rate_at_k
        == 1.0
    )

    assert (
        evaluation_run.summary
        .mean_reciprocal_rank
        == 1.0 / 3
    )

    assert (
        evaluation_run.summary
        .mean_document_coverage
        == 0.5
    )


def test_evaluate_calculates_optimized_metrics(
    db: Session,
) -> None:
    """
    验证Optimized指标计算。
    """

    evaluator = RetrievalEvaluator(
        retrieval_service=FakeRetrievalService(),
    )

    evaluation_run = evaluator.evaluate(
        db=db,
        cases=[
            build_evaluation_case(),
        ],
        retrieval_mode="optimized",
        top_k=2,
        candidate_k=5,
        per_document_limit=1,
    )

    case_result = evaluation_run.cases[0]

    assert case_result.hit is True
    assert case_result.reciprocal_rank == 1.0
    assert case_result.document_coverage == 1.0
    assert case_result.duplicate_rate == 0.0

    assert (
        evaluation_run.summary.hit_rate_at_k
        == 1.0
    )

    assert (
        evaluation_run.summary
        .mean_reciprocal_rank
        == 1.0
    )

    assert (
        evaluation_run.summary
        .mean_document_coverage
        == 1.0
    )

    assert (
        evaluation_run.summary
        .mean_duplicate_rate
        == 0.0
    )


def test_evaluate_returns_zero_when_no_result_matches(
    db: Session,
) -> None:
    """
    验证无召回结果时指标为零。
    """

    evaluator = RetrievalEvaluator(
        retrieval_service=FakeRetrievalService(),
    )

    evaluation_run = evaluator.evaluate(
        db=db,
        cases=[
            build_evaluation_case(
                case_id="no-hit-001",
                question="没有匹配的问题",
            ),
        ],
        retrieval_mode="baseline",
    )

    case_result = evaluation_run.cases[0]

    assert case_result.hit is False
    assert case_result.reciprocal_rank == 0.0
    assert case_result.document_coverage == 0.0
    assert case_result.duplicate_rate == 0.0
    assert case_result.retrieved_document_ids == []
    assert case_result.retrieved_chunk_ids == []


def test_compare_runs_both_retrieval_modes(
    db: Session,
) -> None:
    """
    验证对比评估执行两种模式。
    """

    retrieval_service = FakeRetrievalService()

    evaluator = RetrievalEvaluator(
        retrieval_service=retrieval_service,
    )

    report = evaluator.compare(
        db=db,
        cases=[
            build_evaluation_case(),
        ],
        top_k=3,
        candidate_k=5,
        per_document_limit=1,
    )

    assert retrieval_service.received_modes == [
        "baseline",
        "optimized",
    ]

    assert (
        report.baseline.summary.retrieval_mode
        == "baseline"
    )

    assert (
        report.optimized.summary.retrieval_mode
        == "optimized"
    )

    assert (
        report.baseline.summary
        .mean_document_coverage
        == 0.5
    )

    assert (
        report.optimized.summary
        .mean_document_coverage
        == 1.0
    )


def test_evaluate_rejects_empty_cases(
    db: Session,
) -> None:
    """
    验证空评估集被拒绝。
    """

    evaluator = RetrievalEvaluator(
        retrieval_service=FakeRetrievalService(),
    )

    with pytest.raises(
        ValueError,
        match="evaluation cases cannot be empty",
    ):
        evaluator.evaluate(
            db=db,
            cases=[],
            retrieval_mode="baseline",
        )