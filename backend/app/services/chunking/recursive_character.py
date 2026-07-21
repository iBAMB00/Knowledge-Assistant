from copy import deepcopy
from typing import Any, Sequence

from app.schemas.chunk import ChunkResult
from app.services.chunking.base import ChunkStrategy

RECURSIVE_CHARACTER_STRATEGY_NAME = "recursive_character"

class RecursiveCharacterChunkStrategy(ChunkStrategy):
    """
    递归字符切片策略。

    在 chunk_size 限制范围内，按照分隔符优先级寻找
    最合适的切分位置。

    默认优先级：

        1. 段落
        2. 换行
        3. 中文句号
        4. 中文感叹号
        5. 中文问号
        6. 中文分号
        7. 中文逗号
        8. 空格
        9. 字符硬切

    该策略只负责文本切片，不负责数据库持久化。
    """

    DEFAULT_SEPARATORS: tuple[str, ...] = (
        "\n\n",
        "\n",
        "。",
        "！",
        "？",
        "；",
        "，",
        " ",
        "",
    )

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: Sequence[str] | None = None,
    ) -> None:
        """
        Args:
            chunk_size:
                单个 Chunk 允许包含的最大字符数。

            chunk_overlap:
                相邻 Chunk 之间保留的重叠字符数。

            separators:
                自定义分隔符优先级。
                最后建议保留空字符串，表示无法找到自然分隔点时
                使用字符硬切。
        """

        if chunk_size <= 0:
            raise ValueError(
                "Chunk size must be greater than zero.",
            )

        if chunk_overlap < 0:
            raise ValueError(
                "Chunk overlap cannot be negative.",
            )

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "Chunk overlap must be less than chunk size.",
            )

        resolved_separators = tuple(
            separators
            if separators is not None
            else self.DEFAULT_SEPARATORS
        )

        if not resolved_separators:
            raise ValueError(
                "Separators cannot be empty.",
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = resolved_separators

    @property
    def strategy_name(self) -> str:
        """
        策略唯一名称。

        后续 ChunkService 将通过该名称注册和选择策略。
        """

        return RECURSIVE_CHARACTER_STRATEGY_NAME

    def split(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[ChunkResult]:
        """
        将完整文本切分成多个 ChunkResult。

        ChunkResult 中的 start_offset 和 end_offset
        都使用原始 content 的字符位置，遵循左闭右开区间：

            content[start_offset:end_offset]
        """

        if not content or not content.strip():
            raise ValueError(
                "Content cannot be empty.",
            )

        base_metadata = metadata or {}

        chunks: list[ChunkResult] = []
 
        content_length = len(content)
        start_offset = 0
        chunk_index = 0

        while start_offset < content_length:
            # 跳过前导空格。
            start_offset = self._skip_leading_whitespace(
                content=content,
                start_offset=start_offset,
            )

            if start_offset >= content_length:
                break

            # 计算最大切分位置。
            maximum_end_offset = min(
                start_offset + self.chunk_size,
                content_length,
            )

            # 查找切分位置。
            end_offset = self._find_split_end(
                content=content,
                start_offset=start_offset,
                maximum_end_offset=maximum_end_offset,
                separators=self.separators,
            )

            # 跳过尾随空格。
            end_offset = self._trim_trailing_whitespace(
                content=content,
                start_offset=start_offset,
                end_offset=end_offset,
            )

            if end_offset <= start_offset:
                end_offset = maximum_end_offset

            chunk_content = content[
                start_offset:end_offset
            ]

            chunks.append(
                ChunkResult(
                    content=chunk_content,
                    chunk_index=chunk_index,
                    start_offset=start_offset,
                    end_offset=end_offset,
                    token_count=None,
                    metadata=deepcopy(base_metadata),
                )
            )

            if end_offset >= content_length:
                break

            next_start_offset = (
                end_offset - self.chunk_overlap
            )

            start_offset = self._validate_chunk_progress(
                current_start_offset=start_offset,
                current_end_offset=end_offset,
                next_start_offset=next_start_offset,
            )

            chunk_index += 1

        return chunks

    def _find_split_end(
        self,
        content: str,
        start_offset: int,
        maximum_end_offset: int,
        separators: Sequence[str],
    ) -> int:
        """
        按照分隔符优先级递归寻找切点。

        例如先尝试段落分隔符，如果当前窗口中没有合适段落，
        再尝试换行、句号、逗号，最后执行字符硬切。
        """

        if maximum_end_offset >= len(content):
            return len(content)

        if not separators:
            return maximum_end_offset

        separator = separators[0]

        if separator == "":
            return maximum_end_offset

        # 避免因为靠近开头的分隔符生成非常短的 Chunk
        minimum_end_offset = min(
            start_offset + max(1, self.chunk_size // 2),
            maximum_end_offset,
        )

        # 从后往前查找分隔符。
        separator_position = content.rfind(
            separator,
            minimum_end_offset,
            maximum_end_offset,
        )

        if separator_position == -1:
            return self._find_split_end(
                content=content,
                start_offset=start_offset,
                maximum_end_offset=maximum_end_offset,
                separators=separators[1:],
            )

        # 将分隔符保留在当前 Chunk 中。
        return separator_position + len(separator)

    @staticmethod
    def _skip_leading_whitespace(
        content: str,
        start_offset: int,
    ) -> int:
        """
        跳过 Chunk 开头的空白字符。

        同时移动 offset，保证 offset 仍然对应原文。
        """

        while (
            start_offset < len(content)
            and content[start_offset].isspace()
        ):
            start_offset += 1

        return start_offset

    @staticmethod
    def _trim_trailing_whitespace(
        content: str,
        start_offset: int,
        end_offset: int,
    ) -> int:
        """
        去除 Chunk 末尾多余空白，同时保持原文位置准确。
        """

        while (
            end_offset > start_offset
            and content[end_offset - 1].isspace()
        ):
            end_offset -= 1

        return end_offset
    
    def _validate_chunk_progress(
        self,
        current_start_offset: int,
        current_end_offset: int,
        next_start_offset: int,
    ) -> int:
        """
        校验下一次 Chunk 起始位置是否有效推进。

        防止:
        1. overlap 大于有效内容长度导致重复切片。
        2. 最后短文本不断被重新切分。
        3. offset 无意义递增。

        Args:
            current_start_offset:
                当前 Chunk 起始位置。

            current_end_offset:
                当前 Chunk 结束位置。

            next_start_offset:
                根据 overlap 计算出的下一次起始位置。

        Returns:
            下一次有效切分起始位置。
        """

        chunk_length = (
            current_end_offset
            - current_start_offset
        )

        # 当前chunk长度不足以产生有效overlap，
        # 直接从当前chunk结束位置继续。
        if chunk_length <= self.chunk_overlap:
            return current_end_offset


        # overlap后的起点不能倒退或无效推进。
        if next_start_offset <= current_start_offset:
            return current_end_offset


        return next_start_offset