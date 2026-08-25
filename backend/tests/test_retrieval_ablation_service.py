from datetime import datetime, timezone

import pytest

from app.schemas.retrieval_ablation import (
    RetrievalAblationConfiguration,
    RetrievalAblationEvidenceStrength,
    RetrievalAblationVariant,
)
from app.schemas.retrieval_evaluation import (
    RetrievalCaseCategory,
    RetrievalCaseDifficulty,
    RetrievalEvaluationCase,
    RetrievalEvaluationDatasetReference,
)
from app.schemas.vector_search_result import VectorSearchResult
from app.services.evaluation.retrieval_ablation_service import (
    RetrievalAblationRunner,
    RetrievalAblationVariantRunner,
)
from app.services.evaluation.retrieval_evaluator import RetrievalEvaluator


class SharedEmbedding:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed_query(self, query: str) -> list[float]:
        self.calls.append(query)
        return [float(len(self.calls)), 1.0]


class FakeVariantRetrievalService:
    def __init__(self, *, variant_id: str, embedding: SharedEmbedding) -> None:
        self.variant_id = variant_id
        self.embedding = embedding

    def embed_query(self, query: str) -> list[float]:
        return self.embedding.embed_query(query)

    def retrieve_by_vector(
        self,
        *,
        db,
        query_vector,
        top_k=None,
        candidate_k=None,
        score_threshold=None,
        per_document_limit=None,
        document_id=None,
        retrieval_mode="optimized",
        query_text=None,
    ) -> list[VectorSearchResult]:
        del (
            db,
            query_vector,
            top_k,
            candidate_k,
            score_threshold,
            per_document_limit,
            document_id,
            retrieval_mode,
            query_text,
        )
        correct = VectorSearchResult(
            document_id=1,
            filename="doc.txt",
            chunk_id=10,
            chunk_index=0,
            content="correct",
            score=0.9,
        )
        wrong = VectorSearchResult(
            document_id=1,
            filename="doc.txt",
            chunk_id=11,
            chunk_index=1,
            content="wrong",
            score=0.95,
        )
        if self.variant_id == "full":
            return [correct]
        if self.variant_id == "full_minus_reranker":
            return [wrong, correct]
        if self.variant_id == "full_minus_parent_child":
            return []
        return [wrong]


def _case() -> RetrievalEvaluationCase:
    return RetrievalEvaluationCase(
        case_id="case-1",
        question="where is the correct evidence",
        category=RetrievalCaseCategory.EXACT_TERM,
        difficulty=RetrievalCaseDifficulty.MEDIUM,
        expected_document_ids=[1],
        expected_chunk_ids=[10],
    )


def _dataset() -> RetrievalEvaluationDatasetReference:
    return RetrievalEvaluationDatasetReference(
        schema_version="1.0",
        dataset_id="dataset",
        dataset_version="1.0",
        source_path="evaluation/cases.json",
        source_sha256="a" * 64,
        strict_corpus=True,
        corpus_document_ids=[1],
        total_cases=1,
    )


def _configuration() -> RetrievalAblationConfiguration:
    return RetrievalAblationConfiguration(
        code_version="abc123",
        embedding_provider="fake",
        embedding_model="fake-embedding",
        top_k=2,
        candidate_k=4,
        score_threshold=-1.0,
        per_document_limit=2,
        shared_query_embedding=True,
    )


def _variant(
    variant_id: str,
    *,
    removed_component=None,
    evidence_strength=RetrievalAblationEvidenceStrength.CONTROLLED,
) -> RetrievalAblationVariant:
    return RetrievalAblationVariant(
        variant_id=variant_id,
        label=variant_id,
        retrieval_mode="optimized",
        parent_child_enabled=variant_id != "full_minus_parent_child",
        hybrid_rrf_enabled=True,
        reranker_enabled=variant_id != "full_minus_reranker",
        removed_component=removed_component,
        evidence_strength=evidence_strength,
    )


def test_ablation_reuses_one_query_embedding_and_builds_component_deltas() -> None:
    embedding = SharedEmbedding()
    variants = (
        _variant("full"),
        _variant("full_minus_reranker", removed_component="reranker"),
        _variant(
            "full_minus_parent_child",
            removed_component="parent_child",
            evidence_strength=RetrievalAblationEvidenceStrength.RUNTIME_ONLY,
        ),
    )
    runners = tuple(
        RetrievalAblationVariantRunner(
            variant=variant,
            evaluator=RetrievalEvaluator(
                retrieval_service=FakeVariantRetrievalService(
                    variant_id=variant.variant_id,
                    embedding=embedding,
                )
            ),
        )
        for variant in variants
    )

    report = RetrievalAblationRunner(
        variants=runners,
        full_variant_id="full",
    ).run(
        db=object(),
        cases=[_case()],
        dataset=_dataset(),
        configuration=_configuration(),
    )

    assert embedding.calls == ["where is the correct evidence"]
    by_id = {result.variant.variant_id: result for result in report.variants}
    assert by_id["full"].metrics.chunk_mrr == 1.0
    assert by_id["full_minus_reranker"].metrics.chunk_mrr == 0.5

    reranker_delta = next(
        delta
        for delta in report.deltas
        if delta.variant_id == "full_minus_reranker"
        and delta.metric_id == "chunk_mrr"
    )
    assert reranker_delta.full_advantage == pytest.approx(0.5)
    assert reranker_delta.evidence_strength == RetrievalAblationEvidenceStrength.CONTROLLED
    assert any("Parent-Child" in limitation for limitation in report.limitations)
    assert any("BM25" in limitation and "RRF" in limitation for limitation in report.limitations)
    assert any(
        regression.variant_id == "full_minus_parent_child"
        and regression.case_id == "case-1"
        and "chunk_hit_lower_without_component" in regression.reasons
        for regression in report.case_regressions
    )


def test_ablation_rejects_missing_full_variant() -> None:
    embedding = SharedEmbedding()
    variant = _variant("candidate")
    runner = RetrievalAblationVariantRunner(
        variant=variant,
        evaluator=RetrievalEvaluator(
            retrieval_service=FakeVariantRetrievalService(
                variant_id="candidate",
                embedding=embedding,
            )
        ),
    )
    variant_two = _variant("candidate-2")
    runner_two = RetrievalAblationVariantRunner(
        variant=variant_two,
        evaluator=RetrievalEvaluator(
            retrieval_service=FakeVariantRetrievalService(
                variant_id="candidate-2",
                embedding=embedding,
            )
        ),
    )
    with pytest.raises(ValueError, match="full_variant_id"):
        RetrievalAblationRunner(
            variants=(runner, runner_two),
            full_variant_id="full",
        )


def test_default_ablation_matrix_keeps_hybrid_rrf_as_one_component() -> None:
    from app.services.evaluation.retrieval_ablation_service import (
        build_default_retrieval_ablation_variants,
    )

    variants = build_default_retrieval_ablation_variants()
    by_id = {variant.variant_id: variant for variant in variants}

    assert {
        "legacy_dense_topk",
        "dense_candidate_balance",
        "full",
        "full_minus_hybrid_rrf",
        "full_minus_reranker",
        "full_minus_parent_child",
    } == set(by_id)
    assert by_id["full_minus_hybrid_rrf"].removed_component.value == "hybrid_rrf"
    assert by_id["full_minus_parent_child"].evidence_strength.value == "runtime_only"
    assert not any("bm25" == variant.variant_id for variant in variants)
    assert not any("rrf" == variant.variant_id for variant in variants)
