from datetime import datetime, timezone

import pytest
from sqlalchemy.orm import Session

from app.schemas.retrieval_evaluation import (
    RetrievalCaseCategory,
    RetrievalCaseDifficulty,
    RetrievalEvaluationCase,
    RetrievalEvaluationConfiguration,
    RetrievalEvaluationDatasetReference,
)
from app.schemas.vector_search_result import VectorSearchResult
from app.services.evaluation.retrieval_evaluator import RetrievalEvaluator


class FakeRetrievalService:
    """检索评估测试使用的服务。"""

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
        """根据检索模式返回不同结果。"""

        self.received_modes.append(retrieval_mode)

        if query in {
            "没有匹配的问题",
            "无答案且无召回",
        }:
            return []

        if query == "完整覆盖问题":
            return [
                VectorSearchResult(
                    document_id=2,
                    filename="test-document.txt",
                    chunk_id=10,
                    chunk_index=0,
                    content="目标文档二",
                    score=0.95,
                ),
                VectorSearchResult(
                    document_id=3,
                    filename="test-document.txt",
                    chunk_id=11,
                    chunk_index=0,
                    content="目标文档三",
                    score=0.90,
                ),
            ]

        if query == "部分覆盖问题":
            return [
                VectorSearchResult(
                    document_id=2,
                    filename="test-document.txt",
                    chunk_id=12,
                    chunk_index=0,
                    content="只召回目标文档二",
                    score=0.92,
                ),
            ]

        if retrieval_mode == "baseline":
            return [
                VectorSearchResult(
                    document_id=1,
                    filename="test-document.txt",
                    chunk_id=1,
                    chunk_index=0,
                    content="重复内容",
                    score=0.99,
                ),
                VectorSearchResult(
                    document_id=1,
                    filename="test-document.txt",
                    chunk_id=2,
                    chunk_index=1,
                    content=" 重复内容 ",
                    score=0.98,
                ),
                VectorSearchResult(
                    document_id=2,
                    filename="test-document.txt",
                    chunk_id=3,
                    chunk_index=0,
                    content="预期文档二",
                    score=0.90,
                ),
            ]

        return [
            VectorSearchResult(
                document_id=2,
                filename="test-document.txt",
                chunk_id=3,
                chunk_index=0,
                content="预期文档二",
                score=0.90,
            ),
            VectorSearchResult(
                document_id=3,
                filename="test-document.txt",
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
    """创建测试评估问题。"""

    return RetrievalEvaluationCase(
        case_id=case_id,
        question=question,
        category=RetrievalCaseCategory.MULTI_DOCUMENT,
        difficulty=RetrievalCaseDifficulty.MEDIUM,
        expected_document_ids=[2, 3],
        expected_chunk_ids=[3, 4],
    )


def build_dataset_reference(
) -> RetrievalEvaluationDatasetReference:
    """创建报告数据集快照。"""

    return RetrievalEvaluationDatasetReference(
        schema_version="1.0",
        dataset_id="test-dataset",
        dataset_version="1.0.0",
        source_path="evaluation/test.json",
        source_sha256="0" * 64,
        strict_corpus=True,
        corpus_document_ids=[1, 2, 3],
        total_cases=1,
    )


def build_configuration(
) -> RetrievalEvaluationConfiguration:
    """创建报告运行配置快照。"""

    return RetrievalEvaluationConfiguration(
        executed_at=datetime.now(timezone.utc),
        code_version="abc1234",
        vector_store_backend="database",
        embedding_provider="mock",
        embedding_model="mock-embedding",
        embedding_dimension=3,
        chunk_strategy="recursive_character",
        chunk_size=600,
        chunk_overlap=100,
        top_k=3,
        candidate_k=5,
        score_threshold=-1.0,
        per_document_limit=1,
    )


def test_evaluate_calculates_baseline_metrics(
    db: Session,
) -> None:
    """验证Baseline指标计算。"""

    evaluator = RetrievalEvaluator(
        retrieval_service=FakeRetrievalService(),
    )

    evaluation_run = evaluator.evaluate(
        db=db,
        cases=[build_evaluation_case()],
        retrieval_mode="baseline",
        top_k=3,
    )

    case_result = evaluation_run.cases[0]

    assert case_result.hit is True
    assert case_result.reciprocal_rank == 1.0 / 3
    assert case_result.document_coverage == 0.5
    assert case_result.duplicate_rate == 1.0 / 3
    assert case_result.expected_chunk_ids == [3, 4]
    assert case_result.latency_ms >= 0.0

    assert evaluation_run.summary.total_cases == 1
    assert evaluation_run.summary.answerable_cases == 1
    assert evaluation_run.summary.no_answer_cases == 0
    assert evaluation_run.summary.hit_rate_at_k == 1.0
    assert (
        evaluation_run.summary.mean_reciprocal_rank
        == 1.0 / 3
    )
    assert (
        evaluation_run.summary.mean_document_coverage
        == 0.5
    )
    assert (
        evaluation_run.summary
        .full_document_coverage_rate_at_k
        == 0.0
    )


def test_evaluate_calculates_optimized_metrics(
    db: Session,
) -> None:
    """验证Optimized指标计算。"""

    evaluator = RetrievalEvaluator(
        retrieval_service=FakeRetrievalService(),
    )

    evaluation_run = evaluator.evaluate(
        db=db,
        cases=[build_evaluation_case()],
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

    assert evaluation_run.summary.hit_rate_at_k == 1.0
    assert (
        evaluation_run.summary.mean_reciprocal_rank
        == 1.0
    )
    assert (
        evaluation_run.summary.mean_document_coverage
        == 1.0
    )
    assert (
        evaluation_run.summary
        .full_document_coverage_rate_at_k
        == 1.0
    )


def test_evaluate_returns_zero_when_answerable_case_has_no_result(
    db: Session,
) -> None:
    """验证有答案问题无召回时指标为零。"""

    evaluator = RetrievalEvaluator(
        retrieval_service=FakeRetrievalService(),
    )

    evaluation_run = evaluator.evaluate(
        db=db,
        cases=[
            build_evaluation_case(
                case_id="no-hit-001",
                question="没有匹配的问题",
            )
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


def test_evaluate_handles_no_answer_cases_separately(
    db: Session,
) -> None:
    """验证无答案正确拒绝和误召回不会污染MRR。"""

    evaluator = RetrievalEvaluator(
        retrieval_service=FakeRetrievalService(),
    )

    no_result_case = RetrievalEvaluationCase(
        case_id="no-answer-correct",
        question="无答案且无召回",
        category=RetrievalCaseCategory.NO_ANSWER,
        difficulty=RetrievalCaseDifficulty.EASY,
        should_retrieve=False,
    )
    false_positive_case = RetrievalEvaluationCase(
        case_id="no-answer-false-positive",
        question="无答案但误召回",
        category=RetrievalCaseCategory.NO_ANSWER,
        difficulty=RetrievalCaseDifficulty.MEDIUM,
        should_retrieve=False,
    )

    evaluation_run = evaluator.evaluate(
        db=db,
        cases=[
            build_evaluation_case(),
            no_result_case,
            false_positive_case,
        ],
        retrieval_mode="optimized",
    )

    assert evaluation_run.cases[1].hit is True
    assert (
        evaluation_run.cases[1]
        .no_answer_false_positive
        is False
    )
    assert evaluation_run.cases[2].hit is False
    assert (
        evaluation_run.cases[2]
        .no_answer_false_positive
        is True
    )

    summary = evaluation_run.summary

    assert summary.answerable_cases == 1
    assert summary.no_answer_cases == 2
    assert summary.mean_reciprocal_rank == 1.0
    assert summary.no_answer_accuracy == 0.5
    assert summary.no_answer_false_positive_rate == 0.5
    assert summary.hit_rate_at_k == pytest.approx(2 / 3)


def test_compare_runs_both_modes_and_keeps_run_context(
    db: Session,
) -> None:
    """验证对比报告保留数据集和配置快照。"""

    retrieval_service = FakeRetrievalService()
    evaluator = RetrievalEvaluator(
        retrieval_service=retrieval_service,
    )
    dataset = build_dataset_reference()
    configuration = build_configuration()

    report = evaluator.compare(
        db=db,
        cases=[build_evaluation_case()],
        dataset=dataset,
        configuration=configuration,
        top_k=3,
        candidate_k=5,
        per_document_limit=1,
    )

    assert retrieval_service.received_modes == [
        "baseline",
        "optimized",
    ]
    assert report.dataset == dataset
    assert report.configuration == configuration
    assert (
        report.baseline.summary.retrieval_mode
        == "baseline"
    )
    assert (
        report.optimized.summary.retrieval_mode
        == "optimized"
    )
    assert (
        report.baseline.summary.mean_document_coverage
        == 0.5
    )
    assert (
        report.optimized.summary.mean_document_coverage
        == 1.0
    )


def test_evaluate_rejects_empty_cases(
    db: Session,
) -> None:
    """验证空评估集被拒绝。"""

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


def test_evaluate_distinguishes_mean_and_full_coverage(
    db: Session,
) -> None:
    """验证平均文档覆盖率和严格全覆盖率含义不同。"""

    evaluator = RetrievalEvaluator(
        retrieval_service=FakeRetrievalService(),
    )

    evaluation_run = evaluator.evaluate(
        db=db,
        cases=[
            RetrievalEvaluationCase(
                case_id="full-coverage-001",
                question="完整覆盖问题",
                category=(
                    RetrievalCaseCategory.MULTI_DOCUMENT
                ),
                difficulty=(
                    RetrievalCaseDifficulty.MEDIUM
                ),
                expected_document_ids=[2, 3],
            ),
            RetrievalEvaluationCase(
                case_id="partial-coverage-001",
                question="部分覆盖问题",
                category=(
                    RetrievalCaseCategory.MULTI_DOCUMENT
                ),
                difficulty=(
                    RetrievalCaseDifficulty.MEDIUM
                ),
                expected_document_ids=[2, 3],
            ),
        ],
        retrieval_mode="optimized",
        top_k=2,
    )

    assert (
        evaluation_run.summary.mean_document_coverage
        == pytest.approx(0.75)
    )
    assert (
        evaluation_run.summary
        .full_document_coverage_rate_at_k
        == pytest.approx(0.5)
    )
