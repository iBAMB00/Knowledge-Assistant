import pytest

from app.schemas.vector_search_result import VectorSearchResult
from app.services.rag.context_builder import ContextBuilder


def build_result(
    document_id: int,
    chunk_id: int,
    chunk_index: int,
    content: str,
    score: float,
    filename: str = "test-document.txt",
) -> VectorSearchResult:
    """
    创建测试使用的检索结果。
    """

    return VectorSearchResult(
        document_id=document_id,
        filename=filename,
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

    assert result.sources[0].source_number == 1
    assert result.sources[0].document_id == 1
    assert result.sources[0].chunk_id == 1
    assert result.sources[0].excerpt == "最高相关内容"

    assert result.sources[1].source_number == 2
    assert result.sources[1].excerpt == "中等相关内容"

    assert result.context.index(
        "最高相关内容"
    ) < result.context.index(
        "中等相关内容"
    )

    assert "[来源 1]" in result.context
    assert "[来源 2]" in result.context

    assert "相关度：" not in result.context
    assert "切片ID：" not in result.context

    assert result.sources[0].filename == "test-document.txt"
    assert "文档：test-document.txt" in result.context


def test_build_context_removes_duplicate_content() -> None:
    """
    验证重复内容只保留相关度最高的一条。
    """

    builder = ContextBuilder()

    results = [
        build_result(
            document_id=1,
            filename="test-document.txt",
            chunk_id=1,
            chunk_index=0,
            content="企业知识库检索内容",
            score=0.90,
        ),
        build_result(
            document_id=2,
            filename="test-document.txt",
            chunk_id=2,
            chunk_index=0,
            content=" 企业知识库检索内容 ",
            score=0.80,
        ),
    ]

    result = builder.build(results)

    assert len(result.sources) == 1
    assert result.sources[0].document_id == 1
    assert (
        result.sources[0].excerpt
        == "企业知识库检索内容"
    )

    assert result.context.count(
        "企业知识库检索内容"
    ) == 1


def test_build_context_limits_chunk_count() -> None:
    """
    验证最大 Chunk 数量限制。
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

    original_content = (
        "这是一段非常长的测试内容" * 20
    )

    results = [
        build_result(
            document_id=1,
            chunk_id=1,
            chunk_index=0,
            content=original_content,
            score=0.95,
        )
    ]

    result = builder.build(results)

    assert len(result.context) <= 100
    assert len(result.sources) == 1
    assert "[来源 1]" in result.context

    assert result.sources[0].excerpt in result.context

    assert (
        len(result.sources[0].excerpt)
        < len(original_content)
    )


def test_build_context_source_excerpt_matches_context() -> None:
    """
    验证公开摘要与实际进入上下文的内容一致。
    """

    builder = ContextBuilder(
        default_max_characters=70,
    )

    result = builder.build(
        [
            build_result(
                document_id=1,
                chunk_id=1,
                chunk_index=0,
                content="需要被上下文预算截断的内容" * 10,
                score=0.95,
            )
        ]
    )

    assert len(result.sources) == 1

    source = result.sources[0]

    assert source.excerpt
    assert source.excerpt in result.context

    assert result.context.endswith(
        source.excerpt
    )


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