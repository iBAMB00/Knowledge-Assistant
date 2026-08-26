from collections.abc import Sequence

from app.schemas.vector_search_result import VectorSearchResult
from app.services.reranker.base import (
    RerankItem,
    RerankerCallUsage,
    RerankerProvider,
    RerankerUsageCollector,
    RerankResponse,
)
from app.services.retrieval_service import RetrievalService


class UsageAwareFakeReranker(RerankerProvider):
    @property
    def model_name(self) -> str:
        return "usage-aware-fake"

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_n: int,
    ) -> list[RerankItem]:
        return list(
            self.rerank_with_usage(
                query=query,
                documents=documents,
                top_n=top_n,
            ).items
        )

    def rerank_with_usage(
        self,
        query: str,
        documents: Sequence[str],
        top_n: int,
    ) -> RerankResponse:
        del query, top_n
        return RerankResponse(
            items=(
                RerankItem(index=1, score=0.9),
                RerankItem(index=0, score=0.8),
            ),
            usage=RerankerCallUsage(
                candidate_count=len(documents),
                total_tokens=120,
                token_count_source="provider_usage",
            ),
        )


def _result(chunk_id: int, content: str) -> VectorSearchResult:
    return VectorSearchResult(
        document_id=1,
        filename="doc.txt",
        chunk_id=chunk_id,
        chunk_index=chunk_id,
        content=content,
        score=0.5,
    )


def test_retrieval_service_collects_reranker_provider_usage() -> None:
    collector = RerankerUsageCollector()
    service = RetrievalService(
        embedding_provider=object(),
        vector_store=object(),
        reranker=UsageAwareFakeReranker(),
        reranker_enabled=True,
        reranker_usage_collector=collector,
    )

    reranked = service._rerank_candidates(
        query="哪个更相关？",
        results=[
            _result(1, "候选一"),
            _result(2, "候选二"),
        ],
    )

    assert [item.chunk_id for item in reranked] == [2, 1]

    snapshot = collector.snapshot()
    assert snapshot.request_count == 1
    assert snapshot.successful_request_count == 1
    assert snapshot.failed_request_count == 0
    assert snapshot.candidate_count == 2
    assert snapshot.provider_usage_request_count == 1
    assert snapshot.provider_total_tokens == 120
    assert snapshot.average_candidates_per_request == 2.0
    assert snapshot.average_provider_tokens_per_reported_request == 120.0
    assert snapshot.p95_provider_tokens_per_reported_request == 120.0
    assert snapshot.usage_complete is True


def test_reranker_usage_collector_marks_partial_usage() -> None:
    collector = RerankerUsageCollector()
    collector.record_success(
        RerankerCallUsage(
            candidate_count=3,
            total_tokens=90,
            token_count_source="provider_usage",
        )
    )
    collector.record_success(
        RerankerCallUsage(
            candidate_count=2,
            total_tokens=None,
            token_count_source="unavailable",
        )
    )
    collector.record_failure(candidate_count=4)

    snapshot = collector.snapshot()
    assert snapshot.request_count == 3
    assert snapshot.successful_request_count == 2
    assert snapshot.failed_request_count == 1
    assert snapshot.candidate_count == 9
    assert snapshot.provider_usage_request_count == 1
    assert snapshot.provider_total_tokens == 90
    assert snapshot.usage_complete is False


def test_reranker_usage_collector_preserves_local_tokenizer_source() -> None:
    collector = RerankerUsageCollector()
    collector.record_success(
        RerankerCallUsage(
            candidate_count=3,
            total_tokens=150,
            token_count_source="local_tokenizer",
        )
    )

    snapshot = collector.snapshot()
    assert snapshot.request_count == 1
    assert snapshot.provider_usage_request_count == 0
    assert snapshot.provider_total_tokens is None
    assert snapshot.reported_token_request_count == 1
    assert snapshot.reported_total_tokens == 150
    assert snapshot.average_reported_tokens_per_request == 150.0
    assert snapshot.p95_reported_tokens_per_request == 150.0
    assert snapshot.token_count_source == "local_tokenizer"
    assert snapshot.usage_complete is True
