import math
from collections.abc import Sequence
from time import perf_counter

from sqlalchemy.orm import Session

from app.schemas.retrieval_evaluation import (
    RetrievalComparisonReport,
    RetrievalEvaluationCase,
    RetrievalEvaluationCaseResult,
    RetrievalEvaluationConfiguration,
    RetrievalEvaluationDatasetReference,
    RetrievalEvaluationMode,
    RetrievalEvaluationRun,
    RetrievalEvaluationSummary,
)
from app.schemas.vector_search_result import (
    VectorSearchResult,
)
from app.services.retrieval_service import (
    RetrievalService,
)


class RetrievalEvaluator:
    """
    检索离线评估执行器。

    负责：
    - 使用固定问题集执行检索
    - 分别运行Baseline和Optimized模式
    - 计算单问题评估指标
    - 汇总整体评估结果

    不负责：
    - 创建评估问题集
    - 修改知识库数据
    - 调用大模型生成答案
    - 判断最终回答质量
    """

    def __init__(
        self,
        retrieval_service: RetrievalService,
    ) -> None:
        """初始化检索评估执行器。"""

        self.retrieval_service = retrieval_service

    def compare(
        self,
        db: Session,
        cases: Sequence[RetrievalEvaluationCase],
        dataset: RetrievalEvaluationDatasetReference,
        configuration: RetrievalEvaluationConfiguration,
        top_k: int = 5,
        candidate_k: int = 20,
        score_threshold: float = -1.0,
        per_document_limit: int = 2,
    ) -> RetrievalComparisonReport:
        """使用相同问题集比较两种检索模式。"""

        baseline_run = self.evaluate(
            db=db,
            cases=cases,
            retrieval_mode="baseline",
            top_k=top_k,
            candidate_k=candidate_k,
            score_threshold=score_threshold,
            per_document_limit=per_document_limit,
        )

        optimized_run = self.evaluate(
            db=db,
            cases=cases,
            retrieval_mode="optimized",
            top_k=top_k,
            candidate_k=candidate_k,
            score_threshold=score_threshold,
            per_document_limit=per_document_limit,
        )

        return RetrievalComparisonReport(
            dataset=dataset,
            configuration=configuration,
            baseline=baseline_run,
            optimized=optimized_run,
        )

    def evaluate(
        self,
        db: Session,
        cases: Sequence[RetrievalEvaluationCase],
        retrieval_mode: RetrievalEvaluationMode,
        top_k: int = 5,
        candidate_k: int = 20,
        score_threshold: float = -1.0,
        per_document_limit: int = 2,
    ) -> RetrievalEvaluationRun:
        """执行单种检索模式的完整评估。"""

        if not cases:
            raise ValueError(
                "evaluation cases cannot be empty"
            )

        case_results = [
            self._evaluate_case(
                db=db,
                case=case,
                retrieval_mode=retrieval_mode,
                top_k=top_k,
                candidate_k=candidate_k,
                score_threshold=score_threshold,
                per_document_limit=(
                    per_document_limit
                ),
            )
            for case in cases
        ]

        summary = self._build_summary(
            retrieval_mode=retrieval_mode,
            case_results=case_results,
        )

        return RetrievalEvaluationRun(
            summary=summary,
            cases=case_results,
        )

    def _evaluate_case(
        self,
        db: Session,
        case: RetrievalEvaluationCase,
        retrieval_mode: RetrievalEvaluationMode,
        top_k: int,
        candidate_k: int,
        score_threshold: float,
        per_document_limit: int,
    ) -> RetrievalEvaluationCaseResult:
        """执行并评估单条检索问题。"""

        started_at = perf_counter()

        results = self.retrieval_service.retrieve(
            db=db,
            query=case.question,
            top_k=top_k,
            candidate_k=candidate_k,
            score_threshold=score_threshold,
            per_document_limit=per_document_limit,
            document_id=case.document_id,
            retrieval_mode=retrieval_mode,
        )

        latency_ms = (
            perf_counter() - started_at
        ) * 1000

        expected_document_ids = set(
            case.expected_document_ids
        )

        return RetrievalEvaluationCaseResult(
            case_id=case.case_id,
            question=case.question,
            category=case.category,
            difficulty=case.difficulty,
            should_retrieve=case.should_retrieve,
            retrieval_mode=retrieval_mode,
            expected_document_ids=list(
                case.expected_document_ids
            ),
            expected_chunk_ids=list(
                case.expected_chunk_ids
            ),
            retrieved_document_ids=[
                result.document_id
                for result in results
            ],
            retrieved_chunk_ids=[
                result.chunk_id
                for result in results
            ],
            hit=self._calculate_hit(
                results=results,
                should_retrieve=(
                    case.should_retrieve
                ),
                expected_document_ids=(
                    expected_document_ids
                ),
            ),
            reciprocal_rank=(
                self._calculate_reciprocal_rank(
                    results=results,
                    expected_document_ids=(
                        expected_document_ids
                    ),
                )
                if case.should_retrieve
                else 0.0
            ),
            document_coverage=(
                self._calculate_document_coverage(
                    results=results,
                    expected_document_ids=(
                        expected_document_ids
                    ),
                )
                if case.should_retrieve
                else 0.0
            ),
            duplicate_rate=(
                self._calculate_duplicate_rate(
                    results
                )
            ),
            no_answer_false_positive=(
                not case.should_retrieve
                and bool(results)
            ),
            latency_ms=latency_ms,
        )

    @staticmethod
    def _calculate_hit(
        results: Sequence[VectorSearchResult],
        should_retrieve: bool,
        expected_document_ids: set[int],
    ) -> bool:
        """判断有答案命中或无答案正确拒绝。"""

        if not should_retrieve:
            return not results

        return any(
            result.document_id
            in expected_document_ids
            for result in results
        )

    @staticmethod
    def _calculate_reciprocal_rank(
        results: Sequence[VectorSearchResult],
        expected_document_ids: set[int],
    ) -> float:
        """计算第一个正确结果的倒数排名。"""

        for rank, result in enumerate(
            results,
            start=1,
        ):
            if (
                result.document_id
                in expected_document_ids
            ):
                return 1.0 / rank

        return 0.0

    @staticmethod
    def _calculate_document_coverage(
        results: Sequence[VectorSearchResult],
        expected_document_ids: set[int],
    ) -> float:
        """计算预期文档的召回覆盖比例。"""

        if not expected_document_ids:
            return 0.0

        retrieved_document_ids = {
            result.document_id
            for result in results
        }

        matched_document_ids = (
            expected_document_ids
            & retrieved_document_ids
        )

        return (
            len(matched_document_ids)
            / len(expected_document_ids)
        )

    @staticmethod
    def _calculate_duplicate_rate(
        results: Sequence[VectorSearchResult],
    ) -> float:
        """
        计算重复结果比例。

        相同文档中，标准化后内容完全相同，
        视为重复结果。
        """

        if not results:
            return 0.0

        seen_content_keys: set[
            tuple[int, str]
        ] = set()

        duplicate_count = 0

        for result in results:
            normalized_content = " ".join(
                result.content.split()
            )

            content_key = (
                result.document_id,
                normalized_content,
            )

            if content_key in seen_content_keys:
                duplicate_count += 1
                continue

            seen_content_keys.add(content_key)

        return duplicate_count / len(results)

    @staticmethod
    def _build_summary(
        retrieval_mode: RetrievalEvaluationMode,
        case_results: Sequence[
            RetrievalEvaluationCaseResult
        ],
    ) -> RetrievalEvaluationSummary:
        """汇总所有问题的评估指标。"""

        total_cases = len(case_results)
        answerable_results = [
            result
            for result in case_results
            if result.should_retrieve
        ]
        no_answer_results = [
            result
            for result in case_results
            if not result.should_retrieve
        ]

        answerable_cases = len(
            answerable_results
        )
        no_answer_cases = len(
            no_answer_results
        )

        return RetrievalEvaluationSummary(
            retrieval_mode=retrieval_mode,
            total_cases=total_cases,
            answerable_cases=answerable_cases,
            no_answer_cases=no_answer_cases,
            hit_rate_at_k=(
                sum(
                    1
                    for result in case_results
                    if result.hit
                )
                / total_cases
            ),
            mean_reciprocal_rank=(
                sum(
                    result.reciprocal_rank
                    for result in answerable_results
                )
                / answerable_cases
                if answerable_cases
                else 0.0
            ),
            mean_document_coverage=(
                sum(
                    result.document_coverage
                    for result in answerable_results
                )
                / answerable_cases
                if answerable_cases
                else 0.0
            ),
            full_document_coverage_rate_at_k=(
                sum(
                    1
                    for result in answerable_results
                    if math.isclose(
                        result.document_coverage,
                        1.0,
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    )
                )
                / answerable_cases
                if answerable_cases
                else 0.0
            ),
            mean_duplicate_rate=(
                sum(
                    result.duplicate_rate
                    for result in case_results
                )
                / total_cases
            ),
            no_answer_accuracy=(
                sum(
                    1
                    for result in no_answer_results
                    if result.hit
                )
                / no_answer_cases
                if no_answer_cases
                else 0.0
            ),
            no_answer_false_positive_rate=(
                sum(
                    1
                    for result in no_answer_results
                    if result.no_answer_false_positive
                )
                / no_answer_cases
                if no_answer_cases
                else 0.0
            ),
            average_latency_ms=(
                sum(
                    result.latency_ms
                    for result in case_results
                )
                / total_cases
            ),
        )
