from types import SimpleNamespace

import pytest

from app.schemas.retrieval_evaluation import RetrievalEvaluationDatasetReference
from app.services.evaluation.token_cost_evaluator import (
    LocalEstimatedTokenCounter,
    RetrievalTokenCostEvaluator,
    TokenCostEvaluationOptions,
)
from app.services.reranker.base import (
    RerankerCallUsage,
    RerankerUsageCollector,
)


class FakeDocumentContentRepository:
    def find_by_document_ids(self, db, document_ids):
        return {
            1: SimpleNamespace(id=101, document_id=1, content="甲乙丙丁"),
            2: SimpleNamespace(id=102, document_id=2, content="abcd"),
        }


class FakeDocumentChunkRepository:
    def __init__(self):
        self.chunks = {
            1: SimpleNamespace(id=1, document_content_id=101, content="甲乙丙"),
            2: SimpleNamespace(id=2, document_content_id=101, content="丙丁"),
            3: SimpleNamespace(id=3, document_content_id=102, content="abcd"),
        }

    def find_by_document_content_ids(self, db, document_content_ids):
        return [chunk for chunk in self.chunks.values() if chunk.document_content_id in document_content_ids]

    def find_by_ids(self, db, chunk_ids):
        return [self.chunks[chunk_id] for chunk_id in chunk_ids]


def build_dataset_reference() -> RetrievalEvaluationDatasetReference:
    return RetrievalEvaluationDatasetReference(
        schema_version="1.0",
        dataset_id="token-cost-test",
        dataset_version="1.0.0",
        source_path="evaluation/test.json",
        source_sha256="0" * 64,
        strict_corpus=True,
        corpus_document_ids=[1, 2],
        total_cases=2,
    )


def build_run(case_one_chunks: list[int], case_two_chunks: list[int]):
    return SimpleNamespace(cases=[
        SimpleNamespace(case_id="case-1", question="问题一", retrieved_chunk_ids=case_one_chunks),
        SimpleNamespace(case_id="case-2", question="abc", retrieved_chunk_ids=case_two_chunks),
    ])


def test_local_estimated_token_counter_handles_cjk_ascii_and_symbols():
    counter = LocalEstimatedTokenCounter()
    assert counter.count("你好 abc!") == 4
    assert counter.count("abcdefgh") == 2
    assert counter.count("") == 0


def test_token_cost_evaluator_calculates_ingestion_query_context_and_projection():
    evaluator = RetrievalTokenCostEvaluator(
        document_content_repository=FakeDocumentContentRepository(),
        document_chunk_repository=FakeDocumentChunkRepository(),
    )

    reranker_collector = RerankerUsageCollector()
    reranker_collector.record_success(
        RerankerCallUsage(
            candidate_count=2,
            total_tokens=10,
            token_count_source="provider_usage",
        )
    )
    reranker_collector.record_success(
        RerankerCallUsage(
            candidate_count=3,
            total_tokens=20,
            token_count_source="provider_usage",
        )
    )

    report = evaluator.evaluate(
        db=object(),
        dataset=build_dataset_reference(),
        baseline=build_run([1], [3]),
        optimized=build_run([1, 2], [3]),
        options=TokenCostEvaluationOptions(
            currency="CNY",
            embedding_price_per_million_tokens=1_000_000,
            llm_input_price_per_million_tokens=2_000_000,
            reranker_price_per_million_tokens=1_000_000,
        ),
        optimized_reranker_usage=reranker_collector.snapshot(),
    )

    assert report.token_count_source == "local_estimation"
    assert report.tokenizer_name == "unicode_heuristic_v1"
    assert report.ingestion.document_count == 2
    assert report.ingestion.chunk_count == 3
    assert report.ingestion.source_tokens == 5
    assert report.ingestion.chunk_embedding_tokens == 6
    assert report.ingestion.estimated_overlap_extra_tokens == 1
    assert report.ingestion.estimated_overlap_overhead_rate == pytest.approx(0.2)
    assert report.ingestion.estimated_embedding_cost == pytest.approx(6.0)
    assert report.total_query_embedding_tokens == 4
    assert report.average_query_embedding_tokens == pytest.approx(2.0)
    assert report.estimated_query_embedding_cost == pytest.approx(4.0)
    assert report.baseline.total_context_tokens == 4
    assert report.baseline.estimated_context_input_cost == pytest.approx(8.0)
    assert report.baseline.estimated_retrieval_stage_cost == pytest.approx(12.0)
    assert report.baseline.estimated_average_cost_per_query == pytest.approx(6.0)
    assert report.baseline.estimated_cost_per_1000_queries == pytest.approx(6000.0)
    assert report.baseline.reranker.request_count == 0
    assert report.baseline.reranker.provider_total_tokens == 0
    assert report.baseline.reranker.token_count_source == "not_applicable"
    assert report.optimized.total_context_tokens == 6
    assert report.optimized.reranker.request_count == 2
    assert report.optimized.reranker.candidate_count == 5
    assert report.optimized.reranker.provider_total_tokens == 30
    assert report.optimized.reranker.reported_total_tokens == 30
    assert report.optimized.reranker.usage_complete is True
    assert report.optimized.reranker.reported_provider_token_cost == pytest.approx(30.0)
    assert report.optimized.estimated_retrieval_stage_cost == pytest.approx(46.0)
    assert report.optimized.estimated_average_cost_per_query == pytest.approx(23.0)
    assert report.cases[0].query_tokens == 3
    assert report.cases[0].baseline_context_tokens == 3
    assert report.cases[0].optimized_context_tokens == 5


def test_token_cost_evaluator_rejects_missing_corpus_content():
    class MissingContentRepository(FakeDocumentContentRepository):
        def find_by_document_ids(self, db, document_ids):
            return {1: SimpleNamespace(id=101, document_id=1, content="甲乙")}

    evaluator = RetrievalTokenCostEvaluator(
        document_content_repository=MissingContentRepository(),
        document_chunk_repository=FakeDocumentChunkRepository(),
    )

    with pytest.raises(ValueError, match="missing document contents"):
        evaluator.evaluate(
            db=object(),
            dataset=build_dataset_reference(),
            baseline=build_run([1], [3]),
            optimized=build_run([1], [3]),
            options=TokenCostEvaluationOptions(),
        )


def test_token_cost_evaluator_treats_local_reranker_tokens_as_zero_api_cost():
    evaluator = RetrievalTokenCostEvaluator(
        document_content_repository=FakeDocumentContentRepository(),
        document_chunk_repository=FakeDocumentChunkRepository(),
    )
    collector = RerankerUsageCollector()
    collector.record_success(
        RerankerCallUsage(
            candidate_count=2,
            total_tokens=50,
            token_count_source="local_tokenizer",
        )
    )

    report = evaluator.evaluate(
        db=object(),
        dataset=build_dataset_reference(),
        baseline=build_run([1], [3]),
        optimized=build_run([1, 2], [3]),
        options=TokenCostEvaluationOptions(
            currency="CNY",
            embedding_price_per_million_tokens=0.0,
            llm_input_price_per_million_tokens=0.0,
            reranker_price_per_million_tokens=1000.0,
        ),
        optimized_reranker_usage=collector.snapshot(),
    )

    usage = report.optimized.reranker
    assert usage.reported_total_tokens == 50
    assert usage.provider_total_tokens is None
    assert usage.token_count_source == "local_tokenizer"
    assert usage.reported_provider_token_cost == 0.0
    assert report.optimized.estimated_retrieval_stage_cost == 0.0
