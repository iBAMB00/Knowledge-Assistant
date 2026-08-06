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
    RetrievalEvaluationMetrics,
    RetrievalEvaluationMode,
    RetrievalEvaluationRetrievedResult,
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
    - 让Baseline和Optimized共用同一查询向量
    - 计算文档级、Chunk级和排序指标
    - 记录召回分数与结果摘要
    - 汇总整体、类别和难度指标

    不负责：
    - 创建评估问题集
    - 修改知识库数据
    - 调用大模型生成答案
    - 判断最终回答质量
    """

    CONTENT_EXCERPT_LENGTH = 300

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
        """使用相同问题集和查询向量比较两种检索模式。"""

        if not cases:
            raise ValueError(
                "evaluation cases cannot be empty"
            )

        baseline_results: list[
            RetrievalEvaluationCaseResult
        ] = []
        optimized_results: list[
            RetrievalEvaluationCaseResult
        ] = []

        for case in cases:
            embedding_started_at = perf_counter()
            query_vector = (
                self.retrieval_service.embed_query(
                    case.question
                )
            )
            embedding_latency_ms = (
                perf_counter() - embedding_started_at
            ) * 1000

            baseline_results.append(
                self._evaluate_case_with_vector(
                    db=db,
                    case=case,
                    query_vector=query_vector,
                    embedding_latency_ms=(
                        embedding_latency_ms
                    ),
                    retrieval_mode="baseline",
                    top_k=top_k,
                    candidate_k=candidate_k,
                    score_threshold=score_threshold,
                    per_document_limit=(
                        per_document_limit
                    ),
                )
            )
            optimized_results.append(
                self._evaluate_case_with_vector(
                    db=db,
                    case=case,
                    query_vector=query_vector,
                    embedding_latency_ms=(
                        embedding_latency_ms
                    ),
                    retrieval_mode="optimized",
                    top_k=top_k,
                    candidate_k=candidate_k,
                    score_threshold=score_threshold,
                    per_document_limit=(
                        per_document_limit
                    ),
                )
            )

        baseline_run = self._build_run(
            retrieval_mode="baseline",
            case_results=baseline_results,
        )
        optimized_run = self._build_run(
            retrieval_mode="optimized",
            case_results=optimized_results,
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

        return self._build_run(
            retrieval_mode=retrieval_mode,
            case_results=case_results,
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
        """生成一次查询向量并评估单条问题。"""

        embedding_started_at = perf_counter()
        query_vector = (
            self.retrieval_service.embed_query(
                case.question
            )
        )
        embedding_latency_ms = (
            perf_counter() - embedding_started_at
        ) * 1000

        return self._evaluate_case_with_vector(
            db=db,
            case=case,
            query_vector=query_vector,
            embedding_latency_ms=embedding_latency_ms,
            retrieval_mode=retrieval_mode,
            top_k=top_k,
            candidate_k=candidate_k,
            score_threshold=score_threshold,
            per_document_limit=per_document_limit,
        )

    def _evaluate_case_with_vector(
        self,
        db: Session,
        case: RetrievalEvaluationCase,
        query_vector: Sequence[float],
        embedding_latency_ms: float,
        retrieval_mode: RetrievalEvaluationMode,
        top_k: int,
        candidate_k: int,
        score_threshold: float,
        per_document_limit: int,
    ) -> RetrievalEvaluationCaseResult:
        """使用既有查询向量执行检索并计算指标。"""

        retrieval_started_at = perf_counter()

        results = (
            self.retrieval_service
            .retrieve_by_vector(
                db=db,
                query_vector=query_vector,
                top_k=top_k,
                candidate_k=candidate_k,
                score_threshold=score_threshold,
                per_document_limit=(
                    per_document_limit
                ),
                document_id=case.document_id,
                retrieval_mode=retrieval_mode,
            )
        )

        retrieval_latency_ms = (
            perf_counter() - retrieval_started_at
        ) * 1000

        return self._build_case_result(
            case=case,
            results=results,
            retrieval_mode=retrieval_mode,
            top_k=top_k,
            embedding_latency_ms=embedding_latency_ms,
            retrieval_latency_ms=retrieval_latency_ms,
        )

    def _build_case_result(
        self,
        case: RetrievalEvaluationCase,
        results: Sequence[VectorSearchResult],
        retrieval_mode: RetrievalEvaluationMode,
        top_k: int,
        embedding_latency_ms: float,
        retrieval_latency_ms: float,
    ) -> RetrievalEvaluationCaseResult:
        """根据标准答案和排序结果构造单题报告。"""

        expected_document_ids = set(
            case.expected_document_ids
        )
        expected_chunk_ids = set(
            case.expected_chunk_ids
        )

        document_hit_at_k = (
            self._calculate_document_hit(
                results=results,
                expected_document_ids=(
                    expected_document_ids
                ),
            )
            if case.should_retrieve
            else None
        )

        chunk_metrics_available = (
            case.should_retrieve
            and bool(expected_chunk_ids)
        )

        retrieved_results = [
            self._build_retrieved_result(
                rank=rank,
                result=result,
                expected_document_ids=(
                    expected_document_ids
                ),
                expected_chunk_ids=(
                    expected_chunk_ids
                ),
            )
            for rank, result in enumerate(
                results,
                start=1,
            )
        ]

        first_expected_document_score = (
            self._find_first_expected_score(
                results=results,
                expected_ids=expected_document_ids,
                id_getter=lambda result: (
                    result.document_id
                ),
            )
        )
        first_expected_chunk_score = (
            self._find_first_expected_score(
                results=results,
                expected_ids=expected_chunk_ids,
                id_getter=lambda result: (
                    result.chunk_id
                ),
            )
            if chunk_metrics_available
            else None
        )

        total_latency_ms = (
            embedding_latency_ms
            + retrieval_latency_ms
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
            retrieved_results=retrieved_results,
            hit=(
                bool(document_hit_at_k)
                if case.should_retrieve
                else not results
            ),
            reciprocal_rank=(
                self._calculate_reciprocal_rank(
                    results=results,
                    expected_ids=(
                        expected_document_ids
                    ),
                    id_getter=lambda result: (
                        result.document_id
                    ),
                )
                if case.should_retrieve
                else 0.0
            ),
            document_coverage=(
                self._calculate_recall_at_k(
                    results=results,
                    expected_ids=(
                        expected_document_ids
                    ),
                    id_getter=lambda result: (
                        result.document_id
                    ),
                )
                if case.should_retrieve
                else 0.0
            ),
            document_hit_at_k=document_hit_at_k,
            chunk_hit_at_k=(
                self._calculate_chunk_hit(
                    results=results,
                    expected_chunk_ids=(
                        expected_chunk_ids
                    ),
                )
                if chunk_metrics_available
                else None
            ),
            chunk_reciprocal_rank=(
                self._calculate_reciprocal_rank(
                    results=results,
                    expected_ids=expected_chunk_ids,
                    id_getter=lambda result: (
                        result.chunk_id
                    ),
                )
                if chunk_metrics_available
                else None
            ),
            chunk_recall_at_k=(
                self._calculate_recall_at_k(
                    results=results,
                    expected_ids=expected_chunk_ids,
                    id_getter=lambda result: (
                        result.chunk_id
                    ),
                )
                if chunk_metrics_available
                else None
            ),
            chunk_ndcg_at_k=(
                self._calculate_chunk_ndcg_at_k(
                    results=results,
                    expected_chunk_ids=(
                        expected_chunk_ids
                    ),
                    top_k=top_k,
                )
                if chunk_metrics_available
                else None
            ),
            top_score=(
                results[0].score
                if results
                else None
            ),
            first_expected_document_score=(
                first_expected_document_score
            ),
            first_expected_chunk_score=(
                first_expected_chunk_score
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
            embedding_latency_ms=(
                embedding_latency_ms
            ),
            retrieval_latency_ms=(
                retrieval_latency_ms
            ),
            latency_ms=total_latency_ms,
        )

    @classmethod
    def _build_retrieved_result(
        cls,
        rank: int,
        result: VectorSearchResult,
        expected_document_ids: set[int],
        expected_chunk_ids: set[int],
    ) -> RetrievalEvaluationRetrievedResult:
        """构造带相关性标记和文本摘要的召回明细。"""

        normalized_content = " ".join(
            result.content.split()
        )

        return RetrievalEvaluationRetrievedResult(
            rank=rank,
            document_id=result.document_id,
            filename=result.filename,
            chunk_id=result.chunk_id,
            chunk_index=result.chunk_index,
            score=result.score,
            is_expected_document=(
                result.document_id
                in expected_document_ids
            ),
            is_expected_chunk=(
                result.chunk_id
                in expected_chunk_ids
            ),
            content_excerpt=(
                normalized_content[
                    :cls.CONTENT_EXCERPT_LENGTH
                ]
            ),
        )

    @staticmethod
    def _calculate_document_hit(
        results: Sequence[VectorSearchResult],
        expected_document_ids: set[int],
    ) -> bool:
        """判断Top-K中是否至少命中一个预期文档。"""

        return any(
            result.document_id
            in expected_document_ids
            for result in results
        )

    @staticmethod
    def _calculate_chunk_hit(
        results: Sequence[VectorSearchResult],
        expected_chunk_ids: set[int],
    ) -> bool:
        """判断Top-K中是否至少命中一个预期Chunk。"""

        return any(
            result.chunk_id
            in expected_chunk_ids
            for result in results
        )

    @staticmethod
    def _calculate_reciprocal_rank(
        results: Sequence[VectorSearchResult],
        expected_ids: set[int],
        id_getter,
    ) -> float:
        """计算第一个相关结果的倒数排名。"""

        for rank, result in enumerate(
            results,
            start=1,
        ):
            if id_getter(result) in expected_ids:
                return 1.0 / rank

        return 0.0

    @staticmethod
    def _calculate_recall_at_k(
        results: Sequence[VectorSearchResult],
        expected_ids: set[int],
        id_getter,
    ) -> float:
        """计算Top-K对全部预期ID的覆盖比例。"""

        if not expected_ids:
            return 0.0

        retrieved_ids = {
            id_getter(result)
            for result in results
        }

        return (
            len(expected_ids & retrieved_ids)
            / len(expected_ids)
        )

    @staticmethod
    def _calculate_chunk_ndcg_at_k(
        results: Sequence[VectorSearchResult],
        expected_chunk_ids: set[int],
        top_k: int,
    ) -> float:
        """使用二元Chunk相关性计算nDCG@K。"""

        if not expected_chunk_ids:
            return 0.0

        dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank, result in enumerate(
                results[:top_k],
                start=1,
            )
            if result.chunk_id
            in expected_chunk_ids
        )

        ideal_relevant_count = min(
            len(expected_chunk_ids),
            top_k,
        )
        idcg = sum(
            1.0 / math.log2(rank + 1)
            for rank in range(
                1,
                ideal_relevant_count + 1,
            )
        )

        if idcg == 0.0:
            return 0.0

        return dcg / idcg

    @staticmethod
    def _find_first_expected_score(
        results: Sequence[VectorSearchResult],
        expected_ids: set[int],
        id_getter,
    ) -> float | None:
        """返回排序最靠前的预期结果分数。"""

        for result in results:
            if id_getter(result) in expected_ids:
                return result.score

        return None

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

    @classmethod
    def _build_run(
        cls,
        retrieval_mode: RetrievalEvaluationMode,
        case_results: Sequence[
            RetrievalEvaluationCaseResult
        ],
    ) -> RetrievalEvaluationRun:
        """构造单种模式的完整运行结果。"""

        return RetrievalEvaluationRun(
            summary=cls._build_summary(
                retrieval_mode=retrieval_mode,
                case_results=case_results,
            ),
            cases=list(case_results),
        )

    @classmethod
    def _build_summary(
        cls,
        retrieval_mode: RetrievalEvaluationMode,
        case_results: Sequence[
            RetrievalEvaluationCaseResult
        ],
    ) -> RetrievalEvaluationSummary:
        """汇总整体以及类别、难度分组指标。"""

        metrics = cls._build_metrics(
            case_results
        )

        category_values = sorted({
            result.category.value
            for result in case_results
        })
        difficulty_values = sorted({
            result.difficulty.value
            for result in case_results
        })

        by_category = {
            category: cls._build_metrics([
                result
                for result in case_results
                if result.category.value
                == category
            ])
            for category in category_values
        }
        by_difficulty = {
            difficulty: cls._build_metrics([
                result
                for result in case_results
                if result.difficulty.value
                == difficulty
            ])
            for difficulty in difficulty_values
        }

        return RetrievalEvaluationSummary(
            retrieval_mode=retrieval_mode,
            **metrics.model_dump(),
            by_category=by_category,
            by_difficulty=by_difficulty,
        )

    @classmethod
    def _build_metrics(
        cls,
        case_results: Sequence[
            RetrievalEvaluationCaseResult
        ],
    ) -> RetrievalEvaluationMetrics:
        """计算一组问题的聚合指标。"""

        total_cases = len(case_results)

        if total_cases == 0:
            raise ValueError(
                "case_results cannot be empty"
            )

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
        chunk_labeled_results = [
            result
            for result in answerable_results
            if result.chunk_hit_at_k is not None
        ]

        answerable_cases = len(
            answerable_results
        )
        no_answer_cases = len(
            no_answer_results
        )
        chunk_labeled_cases = len(
            chunk_labeled_results
        )

        first_expected_chunk_scores = [
            result.first_expected_chunk_score
            for result in chunk_labeled_results
            if result.first_expected_chunk_score
            is not None
        ]
        no_answer_false_positive_scores = [
            result.top_score
            for result in no_answer_results
            if (
                result.no_answer_false_positive
                and result.top_score is not None
            )
        ]
        latencies = [
            result.latency_ms
            for result in case_results
        ]

        return RetrievalEvaluationMetrics(
            total_cases=total_cases,
            answerable_cases=answerable_cases,
            no_answer_cases=no_answer_cases,
            chunk_labeled_cases=(
                chunk_labeled_cases
            ),
            hit_rate_at_k=(
                cls._mean([
                    1.0 if result.hit else 0.0
                    for result in case_results
                ])
            ),
            document_hit_rate_at_k=(
                cls._mean([
                    1.0
                    if result.document_hit_at_k
                    else 0.0
                    for result in answerable_results
                ])
                if answerable_cases
                else 0.0
            ),
            mean_reciprocal_rank=(
                cls._mean([
                    result.reciprocal_rank
                    for result in answerable_results
                ])
                if answerable_cases
                else 0.0
            ),
            mean_document_coverage=(
                cls._mean([
                    result.document_coverage
                    for result in answerable_results
                ])
                if answerable_cases
                else 0.0
            ),
            full_document_coverage_rate_at_k=(
                cls._mean([
                    1.0
                    if math.isclose(
                        result.document_coverage,
                        1.0,
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    )
                    else 0.0
                    for result in answerable_results
                ])
                if answerable_cases
                else 0.0
            ),
            chunk_hit_rate_at_k=(
                cls._mean([
                    1.0
                    if result.chunk_hit_at_k
                    else 0.0
                    for result in chunk_labeled_results
                ])
                if chunk_labeled_cases
                else 0.0
            ),
            mean_chunk_reciprocal_rank=(
                cls._mean([
                    result.chunk_reciprocal_rank
                    or 0.0
                    for result in chunk_labeled_results
                ])
                if chunk_labeled_cases
                else 0.0
            ),
            mean_chunk_recall_at_k=(
                cls._mean([
                    result.chunk_recall_at_k
                    or 0.0
                    for result in chunk_labeled_results
                ])
                if chunk_labeled_cases
                else 0.0
            ),
            mean_chunk_ndcg_at_k=(
                cls._mean([
                    result.chunk_ndcg_at_k
                    or 0.0
                    for result in chunk_labeled_results
                ])
                if chunk_labeled_cases
                else 0.0
            ),
            mean_duplicate_rate=cls._mean([
                result.duplicate_rate
                for result in case_results
            ]),
            no_answer_accuracy=(
                cls._mean([
                    1.0 if result.hit else 0.0
                    for result in no_answer_results
                ])
                if no_answer_cases
                else 0.0
            ),
            no_answer_false_positive_rate=(
                cls._mean([
                    1.0
                    if result.no_answer_false_positive
                    else 0.0
                    for result in no_answer_results
                ])
                if no_answer_cases
                else 0.0
            ),
            minimum_first_expected_chunk_score=(
                min(first_expected_chunk_scores)
                if first_expected_chunk_scores
                else None
            ),
            mean_first_expected_chunk_score=(
                cls._mean(
                    first_expected_chunk_scores
                )
                if first_expected_chunk_scores
                else None
            ),
            maximum_no_answer_false_positive_score=(
                max(no_answer_false_positive_scores)
                if no_answer_false_positive_scores
                else None
            ),
            mean_no_answer_false_positive_score=(
                cls._mean(
                    no_answer_false_positive_scores
                )
                if no_answer_false_positive_scores
                else None
            ),
            average_embedding_latency_ms=(
                cls._mean([
                    result.embedding_latency_ms
                    for result in case_results
                ])
            ),
            average_retrieval_latency_ms=(
                cls._mean([
                    result.retrieval_latency_ms
                    for result in case_results
                ])
            ),
            average_latency_ms=cls._mean(
                latencies
            ),
            p50_latency_ms=cls._calculate_percentile(
                values=latencies,
                percentile=0.50,
            ),
            p95_latency_ms=cls._calculate_percentile(
                values=latencies,
                percentile=0.95,
            ),
        )

    @staticmethod
    def _mean(values: Sequence[float]) -> float:
        """计算非空数值序列的算术平均。"""

        if not values:
            raise ValueError(
                "values cannot be empty"
            )

        return sum(values) / len(values)

    @staticmethod
    def _calculate_percentile(
        values: Sequence[float],
        percentile: float,
    ) -> float:
        """使用线性插值计算百分位数。"""

        if not values:
            raise ValueError(
                "values cannot be empty"
            )

        if not 0.0 <= percentile <= 1.0:
            raise ValueError(
                "percentile must be between 0 and 1"
            )

        sorted_values = sorted(values)

        if len(sorted_values) == 1:
            return sorted_values[0]

        position = (
            len(sorted_values) - 1
        ) * percentile
        lower_index = math.floor(position)
        upper_index = math.ceil(position)

        if lower_index == upper_index:
            return sorted_values[lower_index]

        fraction = position - lower_index

        return (
            sorted_values[lower_index]
            + (
                sorted_values[upper_index]
                - sorted_values[lower_index]
            )
            * fraction
        )
