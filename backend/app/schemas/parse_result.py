from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ParsedSection:
    """
    文档结构化章节。

    start_offset / end_offset 基于 ParseResult.content，
    使用左闭右开区间，便于后续结构感知 Chunk 精确回溯原文。
    """

    section_index: int
    title: str | None
    level: int
    heading_path: tuple[str, ...]
    start_offset: int
    end_offset: int

    def to_metadata(self) -> dict[str, Any]:
        """转换为可持久化的 JSON 元数据。"""

        return {
            "section_index": self.section_index,
            "title": self.title,
            "level": self.level,
            "heading_path": list(self.heading_path),
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
        }


@dataclass(frozen=True, slots=True)
class ParseResult:
    """
    文档解析结果。

    content 保存标准化后的全文；sections 保存可选的结构化章节索引。
    结构索引不重复保存章节正文，只保存标题层级和全文偏移，
    避免 document_contents 中出现大段重复文本。
    """

    content: str
    parser_type: str
    parser_version: str
    source_format: str | None = None
    sections: tuple[ParsedSection, ...] = ()

    def to_structure_metadata(self) -> dict[str, Any] | None:
        """生成 DocumentContent 可持久化的结构元数据。"""

        if not self.sections:
            return None

        return {
            "version": "1.0",
            "source_format": self.source_format,
            "sections": [
                section.to_metadata()
                for section in self.sections
            ],
        }
