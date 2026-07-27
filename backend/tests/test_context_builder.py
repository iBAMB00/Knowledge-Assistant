import pytest

from app.schemas.vector_search_result import (
    VectorSearchResult,
)
from app.services.rag.context_builder import ContextBuilder


def build_result(
    document_id: int,
    chunk_id: int,
    chunk_index: int,
    content: str,
    score: float,
) -> VectorSearchResult:
    """
    创建测试使用的检索结果。
    """

    return VectorSearchResult(
        document_id=document_id,
        chunk_id=chunk_id,
        chunk_index=chunk_index,
        content=content,
        score=score,
    )


def test_build_context_orders_results_by_score() -> None:
    """
    验证上下文按照相关度排序并生成来源编号。
    """

    builder = ContextBuilder()

    results = [
        build_result(
            document_id=1,
            chunk_id=2,
            chunk_index=1,
            content="中等相关内容",
            score=0.70,
        ),
        build_result(
            document_id=1,
            chunk_id=1,
            chunk_index=0,
            content="最高相关内容",
            score=0.95,
        ),
    ]

    result = builder.build(results)

    assert len(result.sources) == 2
    assert result.sources[0].chunk_id == 1
    assert result.sources[1].chunk_id == 2

    assert result.context.index(
        "最高相关内容"
    ) < result.context.index(
        "中等相关内容"
    )

    assert "[来源 1]" in result.context
    assert "[来源 2]" in result.context
    assert "相关度：0.9500" in result.context


def test_build_context_removes_duplicate_content() -> None:
    """
    验证重复内容只保留相关度最高的一条。
    """

    builder = ContextBuilder()

    results = [
        build_result(
            document_id=1,
            chunk_id=1,
            chunk_index=0,
            content="企业知识库检索内容",
            score=0.90,
        ),
        build_result(
            document_id=2,
            chunk_id=2,
            chunk_index=0,
            content=" 企业知识库检索内容 ",
            score=0.80,
        ),
    ]

    result = builder.build(results)

    assert len(result.sources) == 1
    assert result.sources[0].chunk_id == 1

    assert result.context.count(
        "企业知识库检索内容"
    ) == 1


def test_build_context_limits_chunk_count() -> None:
    """
    验证最大Chunk数量限制。
    """

    builder = ContextBuilder(
        default_max_chunks=2,
    )

    results = [
        build_result(
            document_id=1,
            chunk_id=index,
            chunk_index=index,
            content=f"检索内容{index}",
            score=1.0 - index * 0.1,
        )
        for index in range(1, 4)
    ]

    result = builder.build(results)

    assert len(result.sources) == 2
    assert "[来源 3]" not in result.context


def test_build_context_limits_character_count() -> None:
    """
    验证上下文字符长度限制。
    """

    builder = ContextBuilder(
        default_max_characters=100,
    )

    results = [
        build_result(
            document_id=1,
            chunk_id=1,
            chunk_index=0,
            content="这是一段非常长的测试内容" * 20,
            score=0.95,
        )
    ]

    result = builder.build(results)

    assert len(result.context) <= 100
    assert len(result.sources) == 1
    assert "[来源 1]" in result.context


def test_build_context_skips_empty_content() -> None:
    """
    验证空内容不会进入上下文。
    """

    builder = ContextBuilder()

    results = [
        build_result(
            document_id=1,
            chunk_id=1,
            chunk_index=0,
            content="   ",
            score=0.95,
        )
    ]

    result = builder.build(results)

    assert result.context == ""
    assert result.sources == []


def test_build_context_returns_empty_result() -> None:
    """
    验证空检索结果可以正常处理。
    """

    builder = ContextBuilder()

    result = builder.build([])

    assert result.context == ""
    assert result.sources == []


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("max_chunks", 0),
        ("max_chunks", True),
        ("max_characters", -1),
        ("max_characters", 1.5),
    ],
)
def test_context_builder_rejects_invalid_parameters(
    field_name: str,
    value: object,
) -> None:
    """
    验证非法上下文参数被拒绝。
    """

    builder = ContextBuilder()

    arguments = {
        field_name: value,
    }

    with pytest.raises(
        ValueError,
        match=field_name,
    ):
        builder.build(
            [],
            **arguments,  # type: ignore[arg-type]
        )