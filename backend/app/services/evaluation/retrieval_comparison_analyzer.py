from collections.abc import Sequence

from app.schemas.retrieval_evaluation import (
    RetrievalComparisonAnalysis,
    RetrievalEvaluationCaseComparison,
    RetrievalEvaluationCaseResult,
    RetrievalEvaluationRun,
    RetrievalRegressionGateCheck,
    RetrievalRegressionGateResult,
    RetrievalRegressionGateThresholds,
)


class RetrievalComparisonAnalyzer:
    """
    Baseline 与候选检索策略的逐题对比分析器。

    负责：
    - 定位 Baseline 未命中正确 Chunk 的问题
    - 定位候选策略相对 Baseline 的 Chunk 退化
    - 定位文档覆盖提升但 Chunk 质量下降的问题
    - 汇总无答案高分误召回问题

    不负责：
    - 执行检索
    - 修改检索策略
    - 判断最终 LLM 回答质量
    """

    FLOAT_EPSILON = 1e-12

    @classmethod
    def analyze(
        cls,
        baseline: RetrievalEvaluationRun,
        optimized: RetrievalEvaluationRun,
    ) -> RetrievalComparisonAnalysis:
        """生成完整逐题失败与退化分析。"""

        baseline_by_case = cls._index_cases(
            baseline.cases
        )
        optimized_by_case = cls._index_cases(
            optimized.cases
        )

        if baseline_by_case.keys() != optimized_by_case.keys():
            raise ValueError(
                "baseline and optimized case IDs must match"
            )

        baseline_chunk_misses: list[
            RetrievalEvaluationCaseComparison
        ] = []
        optimized_regressions: list[
            RetrievalEvaluationCaseComparison
        ] = []
        document_gain_chunk_loss_cases: list[
            RetrievalEvaluationCaseComparison
        ] = []
        no_answer_false_positives: list[
            RetrievalEvaluationCaseComparison
        ] = []

        for case_id, baseline_case in baseline_by_case.items():
            optimized_case = optimized_by_case[case_id]

            if (
                baseline_case.should_retrieve
                and baseline_case.chunk_hit_at_k is False
            ):
                baseline_chunk_misses.append(
                    cls._build_case_comparison(
                        baseline_case=baseline_case,
                        optimized_case=optimized_case,
                        reasons=["baseline_chunk_miss"],
                    )
                )

            regression_reasons = cls._find_regression_reasons(
                baseline_case=baseline_case,
                optimized_case=optimized_case,
            )

            if regression_reasons:
                optimized_regressions.append(
                    cls._build_case_comparison(
                        baseline_case=baseline_case,
                        optimized_case=optimized_case,
                        reasons=regression_reasons,
                    )
                )

            if cls._has_document_gain_with_chunk_loss(
                baseline_case=baseline_case,
                optimized_case=optimized_case,
            ):
                document_gain_chunk_loss_cases.append(
                    cls._build_case_comparison(
                        baseline_case=baseline_case,
                        optimized_case=optimized_case,
                        reasons=[
                            "document_recall_improved",
                            "chunk_recall_decreased",
                        ],
                    )
                )

            if (
                not baseline_case.should_retrieve
                and (
                    baseline_case.no_answer_false_positive
                    or optimized_case.no_answer_false_positive
                )
            ):
                no_answer_false_positives.append(
                    cls._build_case_comparison(
                        baseline_case=baseline_case,
                        optimized_case=optimized_case,
                        reasons=["no_answer_false_positive"],
                    )
                )

        optimized_regressions.sort(
            key=cls._optimized_regression_sort_key
        )
        no_answer_false_positives.sort(
            key=cls._no_answer_false_positive_sort_key
        )

        return RetrievalComparisonAnalysis(
            baseline_chunk_miss_count=len(
                baseline_chunk_misses
            ),
            optimized_regression_count=len(
                optimized_regressions
            ),
            document_gain_chunk_loss_count=len(
                document_gain_chunk_loss_cases
            ),
            no_answer_false_positive_count=len(
                no_answer_false_positives
            ),
            baseline_chunk_misses=baseline_chunk_misses,
            optimized_regressions=optimized_regressions,
            document_gain_chunk_loss_cases=(
                document_gain_chunk_loss_cases
            ),
            no_answer_false_positives=(
                no_answer_false_positives
            ),
        )

    @staticmethod
    def _index_cases(
        cases: Sequence[RetrievalEvaluationCaseResult],
    ) -> dict[str, RetrievalEvaluationCaseResult]:
        """按 case_id 建立唯一索引。"""

        indexed_cases = {
            case.case_id: case
            for case in cases
        }

        if len(indexed_cases) != len(cases):
            raise ValueError(
                "evaluation run contains duplicate case IDs"
            )

        return indexed_cases

    @classmethod
    def _find_regression_reasons(
        cls,
        baseline_case: RetrievalEvaluationCaseResult,
        optimized_case: RetrievalEvaluationCaseResult,
    ) -> list[str]:
        """返回候选策略相对 Baseline 的 Chunk 退化原因。"""

        if not baseline_case.should_retrieve:
            return []

        reasons: list[str] = []

        if (
            baseline_case.chunk_hit_at_k is True
            and optimized_case.chunk_hit_at_k is False
        ):
            reasons.append("chunk_hit_lost")

        if cls._is_lower(
            optimized_case.chunk_recall_at_k,
            baseline_case.chunk_recall_at_k,
        ):
            reasons.append("chunk_recall_decreased")

        if cls._is_lower(
            optimized_case.chunk_reciprocal_rank,
            baseline_case.chunk_reciprocal_rank,
        ):
            reasons.append("chunk_mrr_decreased")

        if cls._is_lower(
            optimized_case.chunk_ndcg_at_k,
            baseline_case.chunk_ndcg_at_k,
        ):
            reasons.append("chunk_ndcg_decreased")

        return reasons

    @classmethod
    def _has_document_gain_with_chunk_loss(
        cls,
        baseline_case: RetrievalEvaluationCaseResult,
        optimized_case: RetrievalEvaluationCaseResult,
    ) -> bool:
        """判断是否出现文档召回提升但 Chunk 召回下降。"""

        if not baseline_case.should_retrieve:
            return False

        return (
            optimized_case.document_coverage
            > baseline_case.document_coverage
            + cls.FLOAT_EPSILON
            and cls._is_lower(
                optimized_case.chunk_recall_at_k,
                baseline_case.chunk_recall_at_k,
            )
        )

    @classmethod
    def _is_lower(
        cls,
        candidate_value: float | None,
        baseline_value: float | None,
    ) -> bool:
        """安全判断候选指标是否发生实质下降。"""

        if (
            candidate_value is None
            or baseline_value is None
        ):
            return False

        return (
            candidate_value
            < baseline_value - cls.FLOAT_EPSILON
        )

    @staticmethod
    def _build_case_comparison(
        baseline_case: RetrievalEvaluationCaseResult,
        optimized_case: RetrievalEvaluationCaseResult,
        reasons: list[str],
    ) -> RetrievalEvaluationCaseComparison:
        """构造可直接写入报告的逐题对比快照。"""

        return RetrievalEvaluationCaseComparison(
            case_id=baseline_case.case_id,
            question=baseline_case.question,
            category=baseline_case.category,
            difficulty=baseline_case.difficulty,
            should_retrieve=baseline_case.should_retrieve,
            reasons=reasons,
            expected_document_ids=(
                baseline_case.expected_document_ids
            ),
            expected_chunk_ids=(
                baseline_case.expected_chunk_ids
            ),
            baseline_document_recall_at_k=(
                baseline_case.document_coverage
            ),
            optimized_document_recall_at_k=(
                optimized_case.document_coverage
            ),
            baseline_chunk_hit_at_k=(
                baseline_case.chunk_hit_at_k
            ),
            optimized_chunk_hit_at_k=(
                optimized_case.chunk_hit_at_k
            ),
            baseline_chunk_recall_at_k=(
                baseline_case.chunk_recall_at_k
            ),
            optimized_chunk_recall_at_k=(
                optimized_case.chunk_recall_at_k
            ),
            baseline_chunk_mrr=(
                baseline_case.chunk_reciprocal_rank
            ),
            optimized_chunk_mrr=(
                optimized_case.chunk_reciprocal_rank
            ),
            baseline_chunk_ndcg_at_k=(
                baseline_case.chunk_ndcg_at_k
            ),
            optimized_chunk_ndcg_at_k=(
                optimized_case.chunk_ndcg_at_k
            ),
            baseline_top_score=baseline_case.top_score,
            optimized_top_score=optimized_case.top_score,
            baseline_retrieved_document_ids=(
                baseline_case.retrieved_document_ids
            ),
            optimized_retrieved_document_ids=(
                optimized_case.retrieved_document_ids
            ),
            baseline_retrieved_chunk_ids=(
                baseline_case.retrieved_chunk_ids
            ),
            optimized_retrieved_chunk_ids=(
                optimized_case.retrieved_chunk_ids
            ),
        )

    @staticmethod
    def _optimized_regression_sort_key(
        comparison: RetrievalEvaluationCaseComparison,
    ) -> tuple[float, float, str]:
        """优先展示 Chunk Recall 和 nDCG 下降最大的用例。"""

        recall_drop = (
            (comparison.baseline_chunk_recall_at_k or 0.0)
            - (comparison.optimized_chunk_recall_at_k or 0.0)
        )
        ndcg_drop = (
            (comparison.baseline_chunk_ndcg_at_k or 0.0)
            - (comparison.optimized_chunk_ndcg_at_k or 0.0)
        )

        return (-recall_drop, -ndcg_drop, comparison.case_id)

    @staticmethod
    def _no_answer_false_positive_sort_key(
        comparison: RetrievalEvaluationCaseComparison,
    ) -> tuple[float, str]:
        """无答案误召回按两种模式的最高 Top1 分数降序排列。"""

        top_score = max(
            (
                comparison.baseline_top_score
                if comparison.baseline_top_score is not None
                else -1.0
            ),
            (
                comparison.optimized_top_score
                if comparison.optimized_top_score is not None
                else -1.0
            ),
        )

        return (-top_score, comparison.case_id)


class RetrievalRegressionGate:
    """
    检索候选策略回归门禁。

    默认规则强调“候选方案至少不能比 Baseline 更差”：
    - 质量指标默认不允许下降
    - 重复率默认不允许增加
    - 平均 Retrieval 和 P95 总延迟允许一定比例波动

    `fail-on-regression` 是否中断进程由 CLI 决定，本类只负责判定。
    """

    FLOAT_EPSILON = 1e-12

    @classmethod
    def evaluate(
        cls,
        baseline: RetrievalEvaluationRun,
        optimized: RetrievalEvaluationRun,
        thresholds: RetrievalRegressionGateThresholds,
    ) -> RetrievalRegressionGateResult:
        """执行全部回归门禁检查。"""

        baseline_summary = baseline.summary
        optimized_summary = optimized.summary

        checks = [
            cls._higher_is_better_check(
                metric="document_hit_rate_at_k",
                baseline_value=baseline_summary.document_hit_rate_at_k,
                optimized_value=optimized_summary.document_hit_rate_at_k,
                allowed_drop=thresholds.max_quality_metric_drop,
            ),
            cls._higher_is_better_check(
                metric="document_mrr",
                baseline_value=baseline_summary.mean_reciprocal_rank,
                optimized_value=optimized_summary.mean_reciprocal_rank,
                allowed_drop=thresholds.max_quality_metric_drop,
            ),
            cls._higher_is_better_check(
                metric="document_recall_at_k",
                baseline_value=baseline_summary.mean_document_coverage,
                optimized_value=optimized_summary.mean_document_coverage,
                allowed_drop=thresholds.max_quality_metric_drop,
            ),
            cls._higher_is_better_check(
                metric="full_document_recall_rate_at_k",
                baseline_value=(
                    baseline_summary.full_document_coverage_rate_at_k
                ),
                optimized_value=(
                    optimized_summary.full_document_coverage_rate_at_k
                ),
                allowed_drop=thresholds.max_quality_metric_drop,
            ),
            cls._higher_is_better_check(
                metric="chunk_hit_rate_at_k",
                baseline_value=baseline_summary.chunk_hit_rate_at_k,
                optimized_value=optimized_summary.chunk_hit_rate_at_k,
                allowed_drop=thresholds.max_quality_metric_drop,
            ),
            cls._higher_is_better_check(
                metric="chunk_mrr",
                baseline_value=baseline_summary.mean_chunk_reciprocal_rank,
                optimized_value=optimized_summary.mean_chunk_reciprocal_rank,
                allowed_drop=thresholds.max_quality_metric_drop,
            ),
            cls._higher_is_better_check(
                metric="chunk_recall_at_k",
                baseline_value=baseline_summary.mean_chunk_recall_at_k,
                optimized_value=optimized_summary.mean_chunk_recall_at_k,
                allowed_drop=thresholds.max_quality_metric_drop,
            ),
            cls._higher_is_better_check(
                metric="chunk_ndcg_at_k",
                baseline_value=baseline_summary.mean_chunk_ndcg_at_k,
                optimized_value=optimized_summary.mean_chunk_ndcg_at_k,
                allowed_drop=thresholds.max_quality_metric_drop,
            ),
            cls._higher_is_better_check(
                metric="no_answer_accuracy",
                baseline_value=baseline_summary.no_answer_accuracy,
                optimized_value=optimized_summary.no_answer_accuracy,
                allowed_drop=thresholds.max_quality_metric_drop,
            ),
            cls._lower_is_better_check(
                metric="duplicate_rate",
                baseline_value=baseline_summary.mean_duplicate_rate,
                optimized_value=optimized_summary.mean_duplicate_rate,
                allowed_increase=(
                    thresholds.max_duplicate_rate_increase
                ),
            ),
            cls._ratio_increase_check(
                metric="average_retrieval_latency_ms",
                baseline_value=(
                    baseline_summary.average_retrieval_latency_ms
                ),
                optimized_value=(
                    optimized_summary.average_retrieval_latency_ms
                ),
                allowed_increase_ratio=(
                    thresholds.max_latency_increase_ratio
                ),
            ),
            cls._ratio_increase_check(
                metric="p95_total_latency_ms",
                baseline_value=baseline_summary.p95_latency_ms,
                optimized_value=optimized_summary.p95_latency_ms,
                allowed_increase_ratio=(
                    thresholds.max_latency_increase_ratio
                ),
            ),
        ]

        failed_metrics = [
            check.metric
            for check in checks
            if not check.passed
        ]

        return RetrievalRegressionGateResult(
            passed=not failed_metrics,
            thresholds=thresholds,
            failed_metrics=failed_metrics,
            checks=checks,
        )

    @classmethod
    def _higher_is_better_check(
        cls,
        metric: str,
        baseline_value: float,
        optimized_value: float,
        allowed_drop: float,
    ) -> RetrievalRegressionGateCheck:
        """构造越高越好的指标门禁。"""

        minimum_allowed = baseline_value - allowed_drop
        passed = (
            optimized_value
            + cls.FLOAT_EPSILON
            >= minimum_allowed
        )

        return RetrievalRegressionGateCheck(
            metric=metric,
            direction="higher_is_better",
            baseline_value=baseline_value,
            optimized_value=optimized_value,
            delta=optimized_value - baseline_value,
            allowed_regression=allowed_drop,
            passed=passed,
        )

    @classmethod
    def _lower_is_better_check(
        cls,
        metric: str,
        baseline_value: float,
        optimized_value: float,
        allowed_increase: float,
    ) -> RetrievalRegressionGateCheck:
        """构造越低越好的绝对值指标门禁。"""

        maximum_allowed = baseline_value + allowed_increase
        passed = (
            optimized_value
            <= maximum_allowed + cls.FLOAT_EPSILON
        )

        return RetrievalRegressionGateCheck(
            metric=metric,
            direction="lower_is_better",
            baseline_value=baseline_value,
            optimized_value=optimized_value,
            delta=optimized_value - baseline_value,
            allowed_regression=allowed_increase,
            passed=passed,
        )

    @classmethod
    def _ratio_increase_check(
        cls,
        metric: str,
        baseline_value: float,
        optimized_value: float,
        allowed_increase_ratio: float,
    ) -> RetrievalRegressionGateCheck:
        """构造允许按 Baseline 比例上涨的延迟门禁。"""

        allowed_increase = (
            baseline_value * allowed_increase_ratio
        )

        return cls._lower_is_better_check(
            metric=metric,
            baseline_value=baseline_value,
            optimized_value=optimized_value,
            allowed_increase=allowed_increase,
        )
