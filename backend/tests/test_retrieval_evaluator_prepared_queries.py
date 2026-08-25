import pytest

from app.schemas.retrieval_evaluation import (
    RetrievalCaseCategory,
    RetrievalCaseDifficulty,
    RetrievalEvaluationCase,
)
from app.schemas.vector_search_result import VectorSearchResult
from app.services.evaluation.retrieval_evaluator import (
    PreparedRetrievalQuery,
    RetrievalEvaluator,
)


class PreparedOnlyRetrievalService:
    def retrieve_by_vector(self, **kwargs):
        assert list(kwargs["query_vector"]) == [1.0, 2.0]
        return [
            VectorSearchResult(
                document_id=1,
                filename="doc.txt",
                chunk_id=10,
                chunk_index=0,
                content="correct",
                score=0.9,
            )
        ]


def _case():
    return RetrievalEvaluationCase(
        case_id="case-1",
        question="q",
        category=RetrievalCaseCategory.EXACT_TERM,
        difficulty=RetrievalCaseDifficulty.EASY,
        expected_document_ids=[1],
        expected_chunk_ids=[10],
    )


def test_evaluate_prepared_uses_frozen_query_vector_without_embedding() -> None:
    evaluator = RetrievalEvaluator(retrieval_service=PreparedOnlyRetrievalService())
    run = evaluator.evaluate_prepared(
        db=object(),
        cases=[_case()],
        prepared_queries={
            "case-1": PreparedRetrievalQuery(
                query_vector=(1.0, 2.0),
                embedding_latency_ms=12.0,
            )
        },
        retrieval_mode="optimized",
        top_k=1,
        candidate_k=2,
        per_document_limit=1,
    )
    assert run.cases[0].hit is True
    assert run.cases[0].embedding_latency_ms == 12.0


def test_evaluate_prepared_rejects_missing_case_vector() -> None:
    evaluator = RetrievalEvaluator(retrieval_service=PreparedOnlyRetrievalService())
    with pytest.raises(ValueError, match="missing evaluation cases"):
        evaluator.evaluate_prepared(
            db=object(),
            cases=[_case()],
            prepared_queries={},
            retrieval_mode="optimized",
        )
