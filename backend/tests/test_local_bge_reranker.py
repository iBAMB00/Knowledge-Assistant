from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from app.services.reranker import local_bge
from app.services.reranker.local_bge import LocalBGERerankerProvider


class FakeScalar:
    def __init__(self, value: int) -> None:
        self.value = value

    def item(self) -> int:
        return self.value


class FakeTensor:
    def __init__(self, values) -> None:
        self.values = values

    def sum(self):
        if isinstance(self.values, list) and self.values and isinstance(self.values[0], list):
            return FakeScalar(sum(sum(row) for row in self.values))
        return FakeScalar(sum(self.values))

    def item(self):
        if isinstance(self.values, list):
            return self.values[0]
        return self.values

    def to(self, device):
        del device
        return self

    def view(self, *shape):
        del shape
        return self

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return list(self.values)


class FakeTokenizer:
    def __init__(self) -> None:
        self.batches: list[list[list[str]]] = []

    def __call__(self, pairs, **kwargs):
        self.batches.append(pairs)
        token_count_per_pair = [5 for _ in pairs]
        return {
            "input_ids": FakeTensor([[1] * 5 for _ in pairs]),
            "attention_mask": FakeTensor(
                [[1] * count for count in token_count_per_pair]
            ),
        }


class FakeModelOutput:
    def __init__(self, scores: list[float]) -> None:
        self.logits = FakeTensor(scores)


class FakeModel:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, **kwargs):
        del kwargs
        score_batches = [
            [0.10, 0.90],
            [0.40],
        ]
        scores = score_batches[self.calls]
        self.calls += 1
        return FakeModelOutput(scores)


class FakeTorch:
    @staticmethod
    def no_grad():
        return nullcontext()


def test_local_bge_reranks_in_batches_and_reports_local_tokenizer_usage(
    monkeypatch,
) -> None:
    tokenizer = FakeTokenizer()
    model = FakeModel()
    monkeypatch.setattr(
        local_bge,
        "_load_local_bge_runtime",
        lambda model_name, device: (FakeTorch(), tokenizer, model),
    )

    provider = LocalBGERerankerProvider(
        model="BAAI/bge-reranker-v2-m3",
        device="cpu",
        batch_size=2,
        max_length=512,
    )
    response = provider.rerank_with_usage(
        query="哪个候选最相关？",
        documents=["候选一", "候选二", "候选三"],
        top_n=2,
    )

    assert [item.index for item in response.items] == [1, 2]
    assert [item.score for item in response.items] == pytest.approx([0.9, 0.4])
    assert len(tokenizer.batches) == 2
    assert response.usage.candidate_count == 3
    assert response.usage.total_tokens == 15
    assert response.usage.token_count_source == "local_tokenizer"


def test_local_bge_empty_documents_return_zero_local_tokens(monkeypatch) -> None:
    monkeypatch.setattr(
        local_bge,
        "_load_local_bge_runtime",
        lambda model_name, device: (FakeTorch(), FakeTokenizer(), FakeModel()),
    )
    provider = LocalBGERerankerProvider()

    response = provider.rerank_with_usage(
        query="测试",
        documents=[],
        top_n=1,
    )

    assert response.items == ()
    assert response.usage.candidate_count == 0
    assert response.usage.total_tokens == 0
    assert response.usage.token_count_source == "local_tokenizer"


def test_local_bge_validates_configuration_before_loading(monkeypatch) -> None:
    load_calls = 0

    def fake_load(model_name, device):
        nonlocal load_calls
        load_calls += 1
        return FakeTorch(), FakeTokenizer(), FakeModel()

    monkeypatch.setattr(local_bge, "_load_local_bge_runtime", fake_load)

    with pytest.raises(ValueError, match="batch_size"):
        LocalBGERerankerProvider(batch_size=0)
    with pytest.raises(ValueError, match="max_length"):
        LocalBGERerankerProvider(max_length=0)
    assert load_calls == 0


def test_reranker_factory_builds_local_bge_without_cloud_credentials(
    monkeypatch,
) -> None:
    from app.services.reranker import factory as factory_module

    captured: dict[str, object] = {}

    class FakeLocalProvider:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    settings = SimpleNamespace(
        reranker_provider="local_bge",
        reranker_model="BAAI/bge-reranker-v2-m3",
        reranker_local_device="cpu",
        reranker_local_batch_size=8,
        reranker_local_max_length=512,
        reranker_api_key=None,
        reranker_base_url=None,
        reranker_timeout=30,
        reranker_instruct=None,
    )
    monkeypatch.setattr(factory_module, "get_settings", lambda: settings)
    monkeypatch.setattr(
        factory_module,
        "LocalBGERerankerProvider",
        FakeLocalProvider,
    )

    provider = factory_module.RerankerFactory.create()

    assert isinstance(provider, FakeLocalProvider)
    assert captured == {
        "model": "BAAI/bge-reranker-v2-m3",
        "device": "cpu",
        "batch_size": 8,
        "max_length": 512,
    }
