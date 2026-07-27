from collections.abc import Sequence
from dataclasses import dataclass

from app.schemas.vector_search_result import VectorSearchResult


@dataclass(frozen=True)
class ContextBuildResult:
    """
    RAG上下文构建结果。
    """

    context: str # 提供给LLM的上下文文本
    sources: list[VectorSearchResult] # 实际进入上下文的检索结果


class ContextBuilder:
    """
    RAG上下文构建器。

    负责：
    - 按相关度整理检索结果
    - 过滤空内容和重复内容
    - 限制上下文Chunk数量
    - 限制上下文最大字符数
    - 为来源生成稳定编号

    不负责：
    - 向量检索
    - Prompt指令设计
    - LLM调用
    """

    def __init__(
        self,
        default_max_chunks: int = 5,
        default_max_characters: int = 6000,
    ) -> None:
        """
        初始化上下文构建器。
        """

        self._validate_positive_integer(
            value=default_max_chunks,
            field_name="max_chunks",
        )
        self._validate_positive_integer(
            value=default_max_characters,
            field_name="max_characters",
        )

        self.default_max_chunks = default_max_chunks
        self.default_max_characters = (
            default_max_characters
        )

    def build(
        self,
        results: Sequence[VectorSearchResult],
        max_chunks: int | None = None,
        max_characters: int | None = None,
    ) -> ContextBuildResult:
        """
        将检索结果构建为模型上下文。
        """

        resolved_max_chunks = (
            self.default_max_chunks
            if max_chunks is None
            else max_chunks
        )

        resolved_max_characters = (
            self.default_max_characters
            if max_characters is None
            else max_characters
        )

        self._validate_positive_integer(
            value=resolved_max_chunks,
            field_name="max_chunks",
        )
        self._validate_positive_integer(
            value=resolved_max_characters,
            field_name="max_characters",
        )

        ordered_results = sorted(
            results,
            key=lambda result: (
                -result.score,
                result.chunk_id,
            ),
        )

        context_blocks: list[str] = []
        selected_sources: list[VectorSearchResult] = []
        seen_content_keys: set[str] = set()
        current_length = 0

        for result in ordered_results:
            if len(selected_sources) >= resolved_max_chunks:
                break

            content = result.content.strip()

            if not content:
                continue

            content_key = self._build_content_key(
                content=content,
            )

            if content_key in seen_content_keys:
                continue

            source_number = len(selected_sources) + 1

            header = self._build_source_header(
                source_number=source_number,
                result=result,
            )

            separator_length = (
                2 if context_blocks else 0
            )

            available_length = (
                resolved_max_characters
                - current_length
                - separator_length
            )

            if available_length <= len(header):
                break

            truncated = False

            if len(header) + len(content) > available_length:
                content = content[
                    : available_length - len(header)
                ].rstrip()

                truncated = True

            if not content:
                break

            block = f"{header}{content}"

            context_blocks.append(block)
            selected_sources.append(result)
            seen_content_keys.add(content_key)

            current_length += (
                separator_length + len(block)
            )

            if truncated:
                break

        return ContextBuildResult(
            context="\n\n".join(context_blocks),
            sources=selected_sources,
        )

    @staticmethod
    def _build_source_header(
        source_number: int,
        result: VectorSearchResult,
    ) -> str:
        """
        构建单个来源的上下文头部。
        """

        return (
            f"[来源 {source_number}]\n"
            f"文档ID：{result.document_id}\n"
            f"切片ID：{result.chunk_id}\n"
            f"切片序号：{result.chunk_index}\n"
            f"相关度：{result.score:.4f}\n"
            "内容："
        )

    @staticmethod
    def _build_content_key(
        content: str,
    ) -> str:
        """
        构建用于去重的标准化内容。
        """

        return " ".join(content.split())

    @staticmethod
    def _validate_positive_integer(
        value: int,
        field_name: str,
    ) -> None:
        """
        校验正整数参数。
        """

        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            raise ValueError(
                f"{field_name} must be "
                "a positive integer"
            )