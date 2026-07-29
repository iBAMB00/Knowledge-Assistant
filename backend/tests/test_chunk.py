import pytest

from app.schemas.chunk import ChunkResult
from app.services.chunking import (
    RecursiveCharacterChunkStrategy,
)
from app.services.chunk_service import (
    ChunkService,
)
from app.services.chunking.factory import (
    ChunkStrategyFactory,
)



# ============================================================
# ChunkResult
# ============================================================


def test_create_chunk_result():
    chunk = ChunkResult(
        content="员工请假需要提前提交申请。",
        chunk_index=0,
        start_offset=0,
        end_offset=14,
        token_count=10,
        metadata={
            "page": 1,
            "section": "请假制度",
        },
    )

    assert chunk.content == "员工请假需要提前提交申请。"
    assert chunk.chunk_index == 0
    assert chunk.start_offset == 0
    assert chunk.end_offset == 14
    assert chunk.token_count == 10
    assert chunk.metadata == {
        "page": 1,
        "section": "请假制度",
    }
    assert chunk.parent_chunk_index is None


def test_chunk_result_rejects_empty_content():
    with pytest.raises(
        ValueError,
        match="Chunk content cannot be empty",
    ):
        ChunkResult(
            content="   ",
            chunk_index=0,
            start_offset=0,
            end_offset=3,
        )


def test_chunk_result_rejects_negative_index():
    with pytest.raises(
        ValueError,
        match="Chunk index cannot be negative",
    ):
        ChunkResult(
            content="hello",
            chunk_index=-1,
            start_offset=0,
            end_offset=5,
        )


def test_chunk_result_rejects_invalid_offsets():
    with pytest.raises(
        ValueError,
        match="Chunk end offset must be greater than start offset",
    ):
        ChunkResult(
            content="hello",
            chunk_index=0,
            start_offset=5,
            end_offset=5,
        )


def test_chunk_result_rejects_negative_token_count():
    with pytest.raises(
        ValueError,
        match="Chunk token count cannot be negative",
    ):
        ChunkResult(
            content="hello",
            chunk_index=0,
            start_offset=0,
            end_offset=5,
            token_count=-1,
        )


def test_chunk_result_metadata_is_not_shared():
    first = ChunkResult(
        content="first",
        chunk_index=0,
        start_offset=0,
        end_offset=5,
    )

    second = ChunkResult(
        content="second",
        chunk_index=1,
        start_offset=5,
        end_offset=11,
    )

    first.metadata["page"] = 1

    assert second.metadata == {}


# ============================================================
# RecursiveCharacterChunkStrategy
# ============================================================


def test_recursive_strategy_name():
    strategy = RecursiveCharacterChunkStrategy()

    assert strategy.strategy_name == "recursive_character"


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap", "error_message"),
    [
        (
            0,
            0,
            "Chunk size must be greater than zero",
        ),
        (
            10,
            -1,
            "Chunk overlap cannot be negative",
        ),
        (
            10,
            10,
            "Chunk overlap must be less than chunk size",
        ),
        (
            10,
            11,
            "Chunk overlap must be less than chunk size",
        ),
    ],
)
def test_recursive_strategy_rejects_invalid_config(
    chunk_size,
    chunk_overlap,
    error_message,
):
    with pytest.raises(
        ValueError,
        match=error_message,
    ):
        RecursiveCharacterChunkStrategy(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )


def test_recursive_strategy_rejects_empty_content():
    strategy = RecursiveCharacterChunkStrategy(
        chunk_size=10,
        chunk_overlap=2,
    )

    with pytest.raises(
        ValueError,
        match="Content cannot be empty",
    ):
        strategy.split("   ")


def test_recursive_strategy_returns_single_chunk_for_short_text():
    content = "员工请假需要提前申请。"

    strategy = RecursiveCharacterChunkStrategy(
        chunk_size=100,
        chunk_overlap=10,
    )

    chunks = strategy.split(
        content=content,
        metadata={
            "filename": "handbook.txt",
        },
    )

    assert len(chunks) == 1

    assert chunks[0].content == content
    assert chunks[0].chunk_index == 0
    assert chunks[0].start_offset == 0
    assert chunks[0].end_offset == len(content)
    assert chunks[0].metadata == {
        "filename": "handbook.txt",
    }


def test_recursive_strategy_uses_hard_split_as_fallback():
    content = "abcdefghij"

    strategy = RecursiveCharacterChunkStrategy(
        chunk_size=6,
        chunk_overlap=2,
        separators=[""],
    )

    chunks = strategy.split(content)

    assert [chunk.content for chunk in chunks] == [
        "abcdef",
        "efghij",
    ]

    assert [
        (
            chunk.start_offset,
            chunk.end_offset,
        )
        for chunk in chunks
    ] == [
        (0, 6),
        (4, 10),
    ]


def test_recursive_strategy_prefers_sentence_boundary():
    content = (
        "员工请假需要提前申请。"
        "主管收到申请后进行审批。"
        "审批通过后请假方可生效。"
    )

    strategy = RecursiveCharacterChunkStrategy(
        chunk_size=20,
        chunk_overlap=0,
    )

    chunks = strategy.split(content)

    assert len(chunks) >= 2

    assert chunks[0].content.endswith("。")

    assert all(
        len(chunk.content) <= 20
        for chunk in chunks
    )


def test_recursive_strategy_preserves_original_offsets():
    content = (
        "第一段内容用于测试。\n\n"
        "第二段内容继续测试。\n\n"
        "第三段内容结束测试。"
    )

    strategy = RecursiveCharacterChunkStrategy(
        chunk_size=15,
        chunk_overlap=3,
    )

    chunks = strategy.split(content)

    for index, chunk in enumerate(chunks):
        assert chunk.chunk_index == index

        assert (
            content[
                chunk.start_offset:chunk.end_offset
            ]
            == chunk.content
        )


def test_recursive_strategy_copies_metadata_for_each_chunk():
    strategy = RecursiveCharacterChunkStrategy(
        chunk_size=6,
        chunk_overlap=2,
        separators=[""],
    )

    chunks = strategy.split(
        content="abcdefghij",
        metadata={
            "filename": "test.txt",
        },
    )

    assert len(chunks) == 2

    chunks[0].metadata["page"] = 1

    assert chunks[1].metadata == {
        "filename": "test.txt",
    }

def test_chunk_service_split():
    """
    测试 ChunkService 能正常调用切片策略。
    """

    service = ChunkService()

    content = (
        "第一段内容。"
        "第二段内容。"
        "第三段内容。"
    )

    chunks = service.split(
        content=content,
        strategy_name="recursive_character",
    )

    assert len(chunks) > 0

    assert chunks[0].content

    assert chunks[0].start_offset == 0


def test_chunk_metadata_isolated():
    """
    测试不同 Chunk 的 metadata 不共享引用。
    """

    service = ChunkService()

    metadata = {
        "document_id": 1
    }

    chunks = service.split(
        content=(
            "第一段内容。"
            "第二段内容。"
        ),
        strategy_name="recursive_character",
        metadata=metadata,
    )


    assert len(chunks) > 0


    if len(chunks) > 1:
        assert (
            chunks[0].metadata
            is not
            chunks[1].metadata
        )

def test_chunk_strategy_factory_uses_settings(
    monkeypatch,
) -> None:
    """
    验证切片策略工厂使用应用配置。
    """

    class FakeSettings:
        """
        测试使用的切片配置。
        """

        chunk_size = 600
        chunk_overlap = 100

    monkeypatch.setattr(
        "app.services.chunking.factory"
        ".get_settings",
        lambda: FakeSettings(),
    )

    strategy = ChunkStrategyFactory.create(
        strategy_name="recursive_character",
    )

    assert isinstance(
        strategy,
        RecursiveCharacterChunkStrategy,
    )

    assert strategy.chunk_size == 600
    assert strategy.chunk_overlap == 100
