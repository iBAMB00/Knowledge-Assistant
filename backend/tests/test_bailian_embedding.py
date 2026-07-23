import math
import os

import pytest

from app.core.config import get_settings
from app.services.embedding.bailian_embedding import (
    BailianEmbeddingProvider,
)


RUN_REAL_TEST = (
    os.getenv(
        "RUN_BAILIAN_EMBEDDING_TEST",
        "",
    ).strip()
    == "1"
)


@pytest.mark.skipif(
    not RUN_REAL_TEST,
    reason=(
        "real Bailian embedding test is disabled; "
        "set RUN_BAILIAN_EMBEDDING_TEST=1 to run"
    ),
)
def test_bailian_embedding_provider_smoke() -> None:
    """
    验证百炼真实Embedding接口。

    验证：
    - 批量文档向量化成功
    - 查询向量化成功
    - 返回数量与输入一致
    - 所有向量维度一致
    - 向量值均为有限数值
    """

    settings = get_settings()

    api_key = _require_setting(
        value=settings.embedding_api_key,
        field_name="embedding_api_key",
    )

    base_url = _require_setting(
        value=settings.embedding_base_url,
        field_name="embedding_base_url",
    )

    model = _require_setting(
        value=settings.embedding_model,
        field_name="embedding_model",
    )

    provider = BailianEmbeddingProvider(
        api_key=api_key,
        base_url=base_url,
        model=model,
        dimension=settings.embedding_dimension,
    )

    documents = [
        "企业知识库用于保存和检索企业内部文档。",
        "RAG通过检索相关知识帮助大模型生成有依据的回答。",
    ]

    document_vectors = provider.embed_documents(
        documents
    )

    query_vector = provider.embed_query(
        "企业知识库有什么作用？"
    )

    assert provider.model_name == model

    assert len(document_vectors) == len(
        documents
    )

    assert all(
        vector
        for vector in document_vectors
    )

    dimensions = {
        len(vector)
        for vector in document_vectors
    }

    assert len(dimensions) == 1

    actual_dimension = next(
        iter(dimensions)
    )

    assert actual_dimension > 0
    assert len(query_vector) == actual_dimension

    if settings.embedding_dimension is not None:
        assert (
            actual_dimension
            == settings.embedding_dimension
        )

    all_vectors = [
        *document_vectors,
        query_vector,
    ]

    assert all(
        isinstance(value, (int, float))
        and math.isfinite(float(value))
        for vector in all_vectors
        for value in vector
    )

    assert (
        document_vectors[0]
        != document_vectors[1]
    )

    print(
        "\nBailian embedding smoke test passed:"
    )
    print(
        f"model={provider.model_name}"
    )
    print(
        f"document_count={len(document_vectors)}"
    )
    print(
        f"dimension={actual_dimension}"
    )


def _require_setting(
    value: str | None,
    field_name: str,
) -> str:
    """
    获取真实接口测试必需配置。
    """

    if value is None or not value.strip():
        pytest.fail(
            f"{field_name} is required "
            "for real Bailian embedding test"
        )

    return value.strip()