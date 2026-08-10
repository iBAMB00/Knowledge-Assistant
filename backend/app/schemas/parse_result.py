from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ParsedPage:
    """
    文档页级定位信息。

    Page 只负责描述正文在原始分页中的位置，不直接决定 Parent Chunk
    边界。start_offset / end_offset 均基于 ParseResult.content。
    """

    page_number: int
    start_offset: int
    end_offset: int
    extraction_method: str

    def to_metadata(self) -> dict[str, Any]:
        """转换为可持久化的页级 JSON 元数据。"""

        return {
            "page_number": self.page_number,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "extraction_method": self.extraction_method,
        }


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
class ParsedBlock:
    """
    文档中的结构化内容块。

    Block 只保存类型、所属 Section、全文偏移和少量结构属性，
    不重复保存代码或表格正文。当前支持 code / table。
    """

    block_index: int
    block_type: str
    section_index: int | None
    start_offset: int
    end_offset: int
    language: str | None = None
    row_count: int | None = None
    column_count: int | None = None
    has_header: bool | None = None

    def to_metadata(self) -> dict[str, Any]:
        """转换为紧凑的可持久化 JSON 元数据。"""

        metadata: dict[str, Any] = {
            "block_index": self.block_index,
            "block_type": self.block_type,
            "section_index": self.section_index,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
        }

        if self.language:
            metadata["language"] = self.language

        if self.row_count is not None:
            metadata["row_count"] = self.row_count

        if self.column_count is not None:
            metadata["column_count"] = self.column_count

        if self.has_header is not None:
            metadata["has_header"] = self.has_header

        return metadata


@dataclass(frozen=True, slots=True)
class ParseResult:
    """
    文档解析结果。

    content 保存标准化后的全文；pages 保存原始分页定位；sections 保存
    章节索引；blocks 保存代码块/表格等局部结构索引。结构数据只保存
    全文偏移和轻量属性，不复制正文，避免 document_contents 中出现
    大段重复文本。
    """

    content: str
    parser_type: str
    parser_version: str
    source_format: str | None = None
    pages: tuple[ParsedPage, ...] = ()
    sections: tuple[ParsedSection, ...] = ()
    blocks: tuple[ParsedBlock, ...] = ()

    def to_structure_metadata(self) -> dict[str, Any] | None:
        """生成 DocumentContent 可持久化的结构元数据。"""

        if not self.pages and not self.sections and not self.blocks:
            return None

        return {
            "version": "1.2",
            "source_format": self.source_format,
            "pages": [
                page.to_metadata()
                for page in self.pages
            ],
            "sections": [
                section.to_metadata()
                for section in self.sections
            ],
            "blocks": [
                block.to_metadata()
                for block in self.blocks
            ],
        }
