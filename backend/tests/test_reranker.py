import json

import pytest

from app.services.reranker.bailian import BailianRerankerProvider


class FakeHTTPResponse:
    """模拟 urllib 响应上下文。"""

    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_bailian_reranker_maps_response_indexes_and_scores(monkeypatch) -> None:
    """验证 qwen3-rerank 响应被转换为统一 RerankItem。"""

    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeHTTPResponse(
            {
                "results": [
                    {"index": 1, "relevance_score": 0.93},
                    {"index": 0, "relevance_score": 0.41},
                ]
            }
        )

    monkeypatch.setattr(
        "app.services.reranker.bailian.urlopen",
        fake_urlopen,
    )

    provider = BailianRerankerProvider(
        api_key="test-key",
        base_url="https://workspace.example/compatible-api/v1/",
        model="qwen3-rerank",
        timeout=15,
    )

    results = provider.rerank(
        query="Java 中什么是空指针异常？",
        documents=["数组越界", "NullPointerException 空指针异常"],
        top_n=2,
    )

    assert [item.index for item in results] == [1, 0]
    assert [item.score for item in results] == pytest.approx([0.93, 0.41])
    assert captured["url"] == "https://workspace.example/compatible-api/v1/reranks"
    assert captured["timeout"] == 15
    assert captured["body"] == {
        "model": "qwen3-rerank",
        "query": "Java 中什么是空指针异常？",
        "documents": ["数组越界", "NullPointerException 空指针异常"],
        "top_n": 2,
    }


def test_bailian_reranker_rejects_invalid_response_index(monkeypatch) -> None:
    """验证服务端返回非法候选索引时显式失败。"""

    monkeypatch.setattr(
        "app.services.reranker.bailian.urlopen",
        lambda request, timeout: FakeHTTPResponse(
            {"results": [{"index": 3, "relevance_score": 0.9}]}
        ),
    )

    provider = BailianRerankerProvider(
        api_key="test-key",
        base_url="https://workspace.example/compatible-api/v1",
    )

    with pytest.raises(RuntimeError, match="out of range"):
        provider.rerank(
            query="测试",
            documents=["候选一", "候选二"],
            top_n=2,
        )
