from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter

from sqlalchemy.orm import Session

from app.repositories.document_chunk_repository import DocumentChunkRepository
from app.schemas.retrieval_ablation import (
    RetrievalAblationComponent,
    RetrievalAblationCaseRegression,
    RetrievalAblationConfiguration,
    RetrievalAblationDelta,
    RetrievalAblationEvidenceStrength,
    RetrievalAblationContextTokenUsage,
    RetrievalAblationMetricSnapshot,
    RetrievalAblationReport,
    RetrievalAblationRerankerTokenUsage,
    RetrievalAblationSharedTokenUsage,
    RetrievalAblationVariant,
    RetrievalAblationVariantResult,
    RetrievalAblationVariantTokenUsage,
)
from app.schemas.retrieval_evaluation import (
    RetrievalEvaluationCase,
    RetrievalEvaluationDatasetReference,
)
from app.services.evaluation.retrieval_evaluator import (
    PreparedRetrievalQuery,
    RetrievalEvaluator,
)
from app.services.evaluation.token_cost_evaluator import LocalEstimatedTokenCounter
from app.services.reranker.base import (
    RerankerUsageCollector,
    RerankerUsageSnapshot,
)


@dataclass(frozen=True)
class RetrievalAblationVariantRunner:
    """一个变体的 Evaluator 与固定实验描述。"""

    variant: RetrievalAblationVariant
    evaluator: RetrievalEvaluator
    reranker_usage_collector: RerankerUsageCollector | None = None


class RetrievalAblationRunner:
    """
    运行固定 Dataset 上的 Retrieval 消融矩阵。

    关键约束：
    - 每个 Case 只生成一次 Query Embedding；所有变体复用同一向量。
    - Full 与 full-minus-X 共用 top_k / candidate_k / threshold 等参数。
    - 不把现有 Hybrid(BM25+RRF) 虚构成两个可独立关闭的生产组件。
    """

    _HIGHER_IS_BETTER = {
        "document_mrr",
        "document_recall_at_k",
        "chunk_hit_rate_at_k",
        "chunk_mrr",
        "chunk_recall_at_k",
        "chunk_ndcg_at_k",
        "no_answer_accuracy",
    }

    def __init__(
        self,
        *,
        variants: Sequence[RetrievalAblationVariantRunner],
        full_variant_id: str,
        document_chunk_repository: DocumentChunkRepository | None = None,
        token_counter: LocalEstimatedTokenCounter | None = None,
    ) -> None:
        if len(variants) < 2:
            raise ValueError("at least two ablation variants are required")
        by_id = {item.variant.variant_id: item for item in variants}
        if len(by_id) != len(variants):
            raise ValueError("ablation variant_id must be unique")
        if full_variant_id not in by_id:
            raise ValueError("full_variant_id must reference a configured variant")

        self.variants = tuple(variants)
        self.full_variant_id = full_variant_id
        self.document_chunk_repository = document_chunk_repository
        self.token_counter = token_counter or LocalEstimatedTokenCounter()

    def run(
        self,
        *,
        db: Session,
        cases: Sequence[RetrievalEvaluationCase],
        dataset: RetrievalEvaluationDatasetReference,
        configuration: RetrievalAblationConfiguration,
    ) -> RetrievalAblationReport:
        if not cases:
            raise ValueError("evaluation cases cannot be empty")

        prepared_queries = self._prepare_queries(cases)
        shared_token_usage = self._build_shared_token_usage(cases)
        results: list[RetrievalAblationVariantResult] = []

        for item in self.variants:
            run = item.evaluator.evaluate_prepared(
                db=db,
                cases=cases,
                prepared_queries=prepared_queries,
                retrieval_mode=item.variant.retrieval_mode,
                top_k=configuration.top_k,
                candidate_k=configuration.candidate_k,
                score_threshold=configuration.score_threshold,
                per_document_limit=configuration.per_document_limit,
            )
            context_usage = self._build_context_token_usage(
                db=db,
                run=run,
            )
            reranker_snapshot = (
                item.reranker_usage_collector.snapshot()
                if item.reranker_usage_collector is not None
                else RerankerUsageSnapshot.empty()
            )
            token_usage = RetrievalAblationVariantTokenUsage(
                reranker=self._build_reranker_token_usage(
                    variant=item.variant,
                    usage=reranker_snapshot,
                    configuration=configuration,
                ),
                final_context=context_usage,
            )
            results.append(
                RetrievalAblationVariantResult(
                    variant=item.variant,
                    metrics=self._snapshot(
                        run=run,
                        context_usage=context_usage,
                    ),
                    token_usage=token_usage,
                    run=run,
                )
            )

        full_result = next(
            result
            for result in results
            if result.variant.variant_id == self.full_variant_id
        )
        deltas = self._build_deltas(full_result, results)
        case_regressions = self._build_case_regressions(full_result, results)
        limitations = self._build_limitations(results)

        return RetrievalAblationReport(
            generated_at=datetime.now(timezone.utc),
            dataset=dataset,
            configuration=configuration,
            full_variant_id=self.full_variant_id,
            shared_token_usage=shared_token_usage,
            variants=tuple(results),
            deltas=tuple(deltas),
            case_regressions=tuple(case_regressions),
            limitations=tuple(limitations),
        )

    def _prepare_queries(
        self,
        cases: Sequence[RetrievalEvaluationCase],
    ) -> dict[str, PreparedRetrievalQuery]:
        preparation_evaluator = self.variants[0].evaluator
        prepared: dict[str, PreparedRetrievalQuery] = {}
        for case in cases:
            started_at = perf_counter()
            query_vector = preparation_evaluator.retrieval_service.embed_query(
                case.question
            )
            prepared[case.case_id] = PreparedRetrievalQuery(
                query_vector=tuple(query_vector),
                embedding_latency_ms=(perf_counter() - started_at) * 1000,
            )
        return prepared

    def _snapshot(
        self,
        *,
        run,
        context_usage: RetrievalAblationContextTokenUsage | None,
    ) -> RetrievalAblationMetricSnapshot:
        summary = run.summary
        return RetrievalAblationMetricSnapshot(
            document_mrr=summary.mean_reciprocal_rank,
            document_recall_at_k=summary.mean_document_coverage,
            chunk_hit_rate_at_k=summary.chunk_hit_rate_at_k,
            chunk_mrr=summary.mean_chunk_reciprocal_rank,
            chunk_recall_at_k=summary.mean_chunk_recall_at_k,
            chunk_ndcg_at_k=summary.mean_chunk_ndcg_at_k,
            no_answer_accuracy=summary.no_answer_accuracy,
            average_context_tokens=(
                context_usage.average_context_tokens
                if context_usage is not None
                else None
            ),
            average_retrieval_latency_ms=summary.average_retrieval_latency_ms,
            p95_total_latency_ms=summary.p95_latency_ms,
        )

    def _build_shared_token_usage(
        self,
        cases: Sequence[RetrievalEvaluationCase],
    ) -> RetrievalAblationSharedTokenUsage:
        query_tokens = [
            self.token_counter.count(case.question)
            for case in cases
        ]
        return RetrievalAblationSharedTokenUsage(
            tokenizer_name=self.token_counter.name,
            request_count=len(cases),
            total_query_embedding_tokens=sum(query_tokens),
            average_query_embedding_tokens=self._mean(query_tokens),
            p95_query_embedding_tokens=self._percentile(
                query_tokens,
                0.95,
            ),
        )

    def _build_context_token_usage(
        self,
        *,
        db: Session,
        run,
    ) -> RetrievalAblationContextTokenUsage | None:
        repository = self.document_chunk_repository
        if repository is None:
            return None

        chunk_ids = sorted(
            {
                chunk_id
                for case in run.cases
                for chunk_id in case.retrieved_chunk_ids
            }
        )
        chunks = repository.find_by_ids(db, chunk_ids) if chunk_ids else []
        chunk_map = {chunk.id: chunk for chunk in chunks}
        if len(chunk_map) != len(chunk_ids):
            return None

        totals = [
            sum(
                self.token_counter.count(chunk_map[chunk_id].content)
                for chunk_id in case.retrieved_chunk_ids
            )
            for case in run.cases
        ]
        return RetrievalAblationContextTokenUsage(
            tokenizer_name=self.token_counter.name,
            total_context_tokens=sum(totals),
            average_context_tokens=self._mean(totals),
            p50_context_tokens=self._percentile(totals, 0.50),
            p95_context_tokens=self._percentile(totals, 0.95),
        )

    @staticmethod
    def _build_reranker_token_usage(
        *,
        variant: RetrievalAblationVariant,
        usage: RerankerUsageSnapshot,
        configuration: RetrievalAblationConfiguration,
    ) -> RetrievalAblationRerankerTokenUsage:
        if not variant.reranker_enabled or usage.request_count == 0:
            source = "not_applicable"
        elif usage.usage_complete:
            source = usage.token_count_source
        elif usage.token_count_source == "provider_usage":
            source = "partial_provider_usage"
        elif usage.token_count_source == "local_tokenizer":
            source = "partial_local_tokenizer"
        elif usage.token_count_source == "mixed":
            source = "partial_mixed"
        else:
            source = "unavailable"

        if usage.provider_total_tokens is not None:
            provider_cost = (
                usage.provider_total_tokens
                / 1_000_000
                * configuration.reranker_price_per_million_tokens
            )
        elif usage.token_count_source == "local_tokenizer":
            provider_cost = 0.0
        else:
            provider_cost = None

        return RetrievalAblationRerankerTokenUsage(
            request_count=usage.request_count,
            successful_request_count=usage.successful_request_count,
            failed_request_count=usage.failed_request_count,
            candidate_count=usage.candidate_count,
            average_candidates_per_request=(
                usage.average_candidates_per_request
            ),
            provider_usage_request_count=(
                usage.provider_usage_request_count
            ),
            provider_total_tokens=usage.provider_total_tokens,
            reported_token_request_count=(
                usage.reported_token_request_count
            ),
            reported_total_tokens=usage.reported_total_tokens,
            average_provider_tokens_per_reported_request=(
                usage.average_provider_tokens_per_reported_request
            ),
            p95_provider_tokens_per_reported_request=(
                usage.p95_provider_tokens_per_reported_request
            ),
            average_reported_tokens_per_request=(
                usage.average_reported_tokens_per_request
            ),
            p95_reported_tokens_per_request=(
                usage.p95_reported_tokens_per_request
            ),
            usage_complete=usage.usage_complete,
            token_count_source=source,
            reported_provider_token_cost=provider_cost,
        )

    @staticmethod
    def _mean(values: Sequence[int]) -> float:
        return sum(values) / len(values) if values else 0.0

    @staticmethod
    def _percentile(
        values: Sequence[int],
        percentile: float,
    ) -> float:
        if not values:
            return 0.0
        sorted_values = sorted(values)
        if len(sorted_values) == 1:
            return float(sorted_values[0])
        position = (len(sorted_values) - 1) * percentile
        lower = int(position)
        upper = min(lower + 1, len(sorted_values) - 1)
        if lower == upper:
            return float(sorted_values[lower])
        fraction = position - lower
        return (
            sorted_values[lower]
            + (sorted_values[upper] - sorted_values[lower])
            * fraction
        )

    def _build_deltas(
        self,
        full_result: RetrievalAblationVariantResult,
        results: Sequence[RetrievalAblationVariantResult],
    ) -> list[RetrievalAblationDelta]:
        deltas: list[RetrievalAblationDelta] = []
        full_metrics = self._metric_mapping(full_result.metrics)

        for result in results:
            if result.variant.variant_id == full_result.variant.variant_id:
                continue
            variant_metrics = self._metric_mapping(result.metrics)
            for metric_id, full_value in full_metrics.items():
                variant_value = variant_metrics.get(metric_id)
                if full_value is None or variant_value is None:
                    continue
                higher = metric_id in self._HIGHER_IS_BETTER
                raw_delta = full_value - variant_value
                full_advantage = raw_delta if higher else -raw_delta
                deltas.append(
                    RetrievalAblationDelta(
                        variant_id=result.variant.variant_id,
                        removed_component=result.variant.removed_component,
                        evidence_strength=result.variant.evidence_strength,
                        metric_id=metric_id,
                        full_value=float(full_value),
                        variant_value=float(variant_value),
                        raw_delta_full_minus_variant=float(raw_delta),
                        full_advantage=float(full_advantage),
                        higher_is_better=higher,
                    )
                )
        return deltas

    def _build_case_regressions(
        self,
        full_result: RetrievalAblationVariantResult,
        results: Sequence[RetrievalAblationVariantResult],
    ) -> list[RetrievalAblationCaseRegression]:
        regressions: list[RetrievalAblationCaseRegression] = []
        full_cases = {case.case_id: case for case in full_result.run.cases}
        epsilon = 1e-12

        for result in results:
            if result.variant.variant_id == full_result.variant.variant_id:
                continue
            variant_cases = {case.case_id: case for case in result.run.cases}
            if full_cases.keys() != variant_cases.keys():
                raise ValueError("ablation variants must evaluate identical case IDs")

            for case_id, full_case in full_cases.items():
                variant_case = variant_cases[case_id]
                reasons: list[str] = []
                if full_case.hit and not variant_case.hit:
                    reasons.append("case_success_lower_without_component")
                if (
                    full_case.chunk_hit_at_k is True
                    and variant_case.chunk_hit_at_k is False
                ):
                    reasons.append("chunk_hit_lower_without_component")
                if (
                    full_case.chunk_recall_at_k is not None
                    and variant_case.chunk_recall_at_k is not None
                    and full_case.chunk_recall_at_k
                    > variant_case.chunk_recall_at_k + epsilon
                ):
                    reasons.append("chunk_recall_lower_without_component")
                if (
                    full_case.chunk_ndcg_at_k is not None
                    and variant_case.chunk_ndcg_at_k is not None
                    and full_case.chunk_ndcg_at_k
                    > variant_case.chunk_ndcg_at_k + epsilon
                ):
                    reasons.append("chunk_ndcg_lower_without_component")

                if not reasons:
                    continue
                regressions.append(
                    RetrievalAblationCaseRegression(
                        variant_id=result.variant.variant_id,
                        case_id=case_id,
                        removed_component=result.variant.removed_component,
                        evidence_strength=result.variant.evidence_strength,
                        reasons=tuple(reasons),
                        full_chunk_hit=full_case.chunk_hit_at_k,
                        variant_chunk_hit=variant_case.chunk_hit_at_k,
                        full_chunk_recall_at_k=full_case.chunk_recall_at_k,
                        variant_chunk_recall_at_k=variant_case.chunk_recall_at_k,
                        full_chunk_ndcg_at_k=full_case.chunk_ndcg_at_k,
                        variant_chunk_ndcg_at_k=variant_case.chunk_ndcg_at_k,
                    )
                )

        return regressions

    @staticmethod
    def _metric_mapping(
        snapshot: RetrievalAblationMetricSnapshot,
    ) -> Mapping[str, float | None]:
        return {
            "document_mrr": snapshot.document_mrr,
            "document_recall_at_k": snapshot.document_recall_at_k,
            "chunk_hit_rate_at_k": snapshot.chunk_hit_rate_at_k,
            "chunk_mrr": snapshot.chunk_mrr,
            "chunk_recall_at_k": snapshot.chunk_recall_at_k,
            "chunk_ndcg_at_k": snapshot.chunk_ndcg_at_k,
            "no_answer_accuracy": snapshot.no_answer_accuracy,
            "average_context_tokens": snapshot.average_context_tokens,
            "average_retrieval_latency_ms": snapshot.average_retrieval_latency_ms,
            "p95_total_latency_ms": snapshot.p95_total_latency_ms,
        }

    @staticmethod
    def _build_limitations(
        results: Sequence[RetrievalAblationVariantResult],
    ) -> list[str]:
        limitations = [
            "BM25 and RRF are coupled by the current production Hybrid path; "
            "this report attributes them as one hybrid_rrf component and does not "
            "claim separate BM25-versus-RRF causal contribution.",
            "Reference variants run against the current persisted corpus/index; "
            "they are comparison anchors, not guaranteed reproductions of the "
            "historical pre-optimization baseline.",
            "Latency values are single-run wall-clock observations. Use repeated "
            "runs before making statistical latency claims."
        ]
        if any(
            result.variant.evidence_strength
            == RetrievalAblationEvidenceStrength.RUNTIME_ONLY
            for result in results
        ):
            limitations.append(
                "Parent-Child removal is a runtime ablation on the existing index. "
                "Because the current index can already contain both Parent and Child "
                "chunks, it is not equivalent to rebuilding a corpus without "
                "Parent-Child ingestion."
            )
        return limitations


def build_default_retrieval_ablation_variants(
    *,
    include_parent_child_runtime_ablation: bool = True,
) -> tuple[RetrievalAblationVariant, ...]:
    """返回 A7 冻结的默认消融矩阵。"""

    variants: list[RetrievalAblationVariant] = [
        RetrievalAblationVariant(
            variant_id="legacy_dense_topk",
            label="Legacy-mode Dense Top-K reference (current index)",
            retrieval_mode="baseline",
            parent_child_enabled=False,
            hybrid_rrf_enabled=False,
            reranker_enabled=False,
            evidence_strength=RetrievalAblationEvidenceStrength.REFERENCE,
            notes=(
                "Legacy retrieval mode on the current persisted index; this is "
                "a reference anchor, not a reconstruction of historical evidence."
            ),
        ),
        RetrievalAblationVariant(
            variant_id="dense_candidate_balance",
            label="Dense candidate expansion + document balance",
            retrieval_mode="optimized",
            parent_child_enabled=False,
            hybrid_rrf_enabled=False,
            reranker_enabled=False,
            evidence_strength=RetrievalAblationEvidenceStrength.REFERENCE,
            notes=(
                "Optimized pipeline reference with all three advanced "
                "components disabled."
            ),
        ),
        RetrievalAblationVariant(
            variant_id="full",
            label="Full Retrieval pipeline",
            retrieval_mode="optimized",
            parent_child_enabled=True,
            hybrid_rrf_enabled=True,
            reranker_enabled=True,
            evidence_strength=RetrievalAblationEvidenceStrength.CONTROLLED,
        ),
        RetrievalAblationVariant(
            variant_id="full_minus_hybrid_rrf",
            label="Full minus Hybrid(BM25+RRF)",
            retrieval_mode="optimized",
            parent_child_enabled=True,
            hybrid_rrf_enabled=False,
            reranker_enabled=True,
            removed_component=RetrievalAblationComponent.HYBRID_RRF,
            evidence_strength=RetrievalAblationEvidenceStrength.CONTROLLED,
        ),
        RetrievalAblationVariant(
            variant_id="full_minus_reranker",
            label="Full minus Reranker",
            retrieval_mode="optimized",
            parent_child_enabled=True,
            hybrid_rrf_enabled=True,
            reranker_enabled=False,
            removed_component=RetrievalAblationComponent.RERANKER,
            evidence_strength=RetrievalAblationEvidenceStrength.CONTROLLED,
        ),
    ]
    if include_parent_child_runtime_ablation:
        variants.append(
            RetrievalAblationVariant(
                variant_id="full_minus_parent_child",
                label="Full minus Parent-Child runtime behavior",
                retrieval_mode="optimized",
                parent_child_enabled=False,
                hybrid_rrf_enabled=True,
                reranker_enabled=True,
                removed_component=RetrievalAblationComponent.PARENT_CHILD,
                evidence_strength=RetrievalAblationEvidenceStrength.RUNTIME_ONLY,
                notes=(
                    "Runtime-only ablation on the existing index; the index may "
                    "already contain both Parent and Child chunks."
                ),
            )
        )
    return tuple(variants)
