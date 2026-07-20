import pytest

from app.schemas.chunk import ChunkResult


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