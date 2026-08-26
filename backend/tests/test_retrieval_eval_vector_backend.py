from argparse import Namespace
from types import SimpleNamespace

import scripts.run_retrieval_ablation as ablation_script
import scripts.run_retrieval_evaluation as evaluation_script


class FakeEmbeddingProvider:
    model_name = "fake-embedding"


def _settings(backend: str = "qdrant") -> SimpleNamespace:
    return SimpleNamespace(
        vector_store_backend=backend,
        embedding_provider="fake",
        embedding_dimension=1024,
        retrieval_top_k=5,
        retrieval_candidate_k=20,
        retrieval_score_threshold=-1.0,
        retrieval_per_document_limit=2,
        retrieval_rrf_k=60,
        retrieval_hybrid_enabled=True,
        parent_child_enabled=True,
        parent_child_child_size=300,
        parent_child_child_overlap=50,
        reranker_enabled=True,
        reranker_model="fake-reranker",
        reranker_fail_open=False,
        chunk_strategy="recursive_character",
        chunk_size=600,
        chunk_overlap=100,
    )


def _args() -> Namespace:
    return Namespace(
        code_version="abc1234",
        top_k=5,
        candidate_k=20,
        score_threshold=-1.0,
        per_document_limit=2,
    )


def test_retrieval_evaluation_uses_vector_store_factory_and_records_backend(
    monkeypatch,
) -> None:
    fake_store = object()
    fake_settings = _settings("qdrant")

    monkeypatch.setattr(evaluation_script, "settings", fake_settings)
    monkeypatch.setattr(
        evaluation_script.EmbeddingFactory,
        "create",
        staticmethod(lambda: FakeEmbeddingProvider()),
    )
    monkeypatch.setattr(
        evaluation_script.VectorStoreFactory,
        "create",
        staticmethod(
            lambda settings=None: SimpleNamespace(
                vector_store=fake_store,
                vector_index=fake_store,
            )
        ),
    )
    monkeypatch.setattr(
        evaluation_script.RerankerFactory,
        "create",
        staticmethod(lambda: object()),
    )

    components = evaluation_script.build_retrieval_evaluation_components()

    assert (
        components.evaluator.retrieval_service.vector_store
        is fake_store
    )

    configuration = evaluation_script.build_configuration(
        args=_args(),
        embedding_model=components.embedding_model,
    )

    assert configuration.vector_store_backend == "qdrant"


def test_retrieval_ablation_uses_vector_store_factory_and_records_backend(
    monkeypatch,
) -> None:
    fake_store = object()
    fake_settings = _settings("qdrant")

    monkeypatch.setattr(ablation_script, "settings", fake_settings)
    monkeypatch.setattr(
        ablation_script.EmbeddingFactory,
        "create",
        staticmethod(lambda: FakeEmbeddingProvider()),
    )
    monkeypatch.setattr(
        ablation_script.VectorStoreFactory,
        "create",
        staticmethod(
            lambda settings=None: SimpleNamespace(
                vector_store=fake_store,
                vector_index=fake_store,
            )
        ),
    )
    monkeypatch.setattr(
        ablation_script.RerankerFactory,
        "create",
        staticmethod(lambda: object()),
    )

    shared = ablation_script.build_shared_dependencies()

    assert shared.vector_store is fake_store

    configuration = ablation_script.build_configuration(
        args=_args(),
        shared=shared,
        code_version="abc1234",
    )

    assert configuration.vector_store_backend == "qdrant"
