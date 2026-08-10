from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from app.schemas.chunk import ChunkResult
from app.services.chunking.base import ChunkStrategy


@dataclass(frozen=True, slots=True)
class _SectionBoundary:
    """经过校验的章节边界。"""

    section_index: int
    title: str | None
    level: int
    heading_path: tuple[str, ...]
    start_offset: int
    end_offset: int


class SectionAwareParentChunker:
    """
    基于文档章节边界生成 Parent Chunk。

    章节只负责限制 Parent 的语义边界；章节过长时仍委托现有
    ChunkStrategy 在章节内部切分，避免重复实现长度与 overlap 算法。
    """

    def __init__(self, base_strategy: ChunkStrategy) -> None:
        self.base_strategy = base_strategy

    def split(
        self,
        content: str,
        structure_metadata: dict[str, Any] | None,
        metadata: dict[str, Any] | None = None,
    ) -> list[ChunkResult]:
        """
        按 Section 生成 Parent Chunk。

        结构元数据不完整或与正文不一致时返回空列表，
        由上层安全降级到普通全文切片，避免部分正文被静默遗漏。
        """

        sections = self._normalize_sections(
            content=content,
            structure_metadata=structure_metadata,
        )

        if not sections:
            return []

        source_format = (
            structure_metadata.get("source_format")
            if isinstance(structure_metadata, dict)
            else None
        )
        base_metadata = metadata or {}
        results: list[ChunkResult] = []
        next_chunk_index = 0

        for section in sections:
            section_content = content[
                section.start_offset:section.end_offset
            ]
            section_metadata = deepcopy(base_metadata)
            section_metadata.update(
                {
                    "structure_aware": True,
                    "chunk_boundary_mode": "section",
                    "source_format": source_format,
                    "section_index": section.section_index,
                    "section_title": section.title,
                    "section_level": section.level,
                    "heading_path": list(section.heading_path),
                    "section_start_offset": section.start_offset,
                    "section_end_offset": section.end_offset,
                }
            )

            section_chunks = self.base_strategy.split(
                content=section_content,
                metadata=section_metadata,
            )
            section_part_count = len(section_chunks)

            for section_part_index, section_chunk in enumerate(
                section_chunks
            ):
                document_start_offset = (
                    section.start_offset
                    + section_chunk.start_offset
                )
                document_end_offset = (
                    section.start_offset
                    + section_chunk.end_offset
                )
                chunk_metadata = deepcopy(
                    section_chunk.metadata
                )
                chunk_metadata.update(
                    {
                        "section_part_index": section_part_index,
                        "section_part_count": section_part_count,
                        "document_start_offset": document_start_offset,
                        "document_end_offset": document_end_offset,
                    }
                )

                results.append(
                    ChunkResult(
                        content=content[
                            document_start_offset:document_end_offset
                        ],
                        chunk_index=next_chunk_index,
                        start_offset=document_start_offset,
                        end_offset=document_end_offset,
                        token_count=section_chunk.token_count,
                        metadata=chunk_metadata,
                    )
                )
                next_chunk_index += 1

        return results

    @classmethod
    def _normalize_sections(
        cls,
        content: str,
        structure_metadata: dict[str, Any] | None,
    ) -> tuple[_SectionBoundary, ...]:
        """校验并标准化持久化章节索引。"""

        if not isinstance(structure_metadata, dict):
            return ()

        raw_sections = structure_metadata.get("sections")

        if not isinstance(raw_sections, list) or not raw_sections:
            return ()

        sections: list[_SectionBoundary] = []

        for position, raw_section in enumerate(raw_sections):
            if not isinstance(raw_section, dict):
                return ()

            start_offset = raw_section.get("start_offset")
            end_offset = raw_section.get("end_offset")
            level = raw_section.get("level", 0)
            section_index = raw_section.get(
                "section_index",
                position,
            )
            title = raw_section.get("title")
            heading_path = raw_section.get(
                "heading_path",
                [],
            )

            if (
                not cls._is_integer(start_offset)
                or not cls._is_integer(end_offset)
                or not cls._is_integer(level)
                or not cls._is_integer(section_index)
            ):
                return ()

            if (
                start_offset < 0
                or end_offset <= start_offset
                or end_offset > len(content)
                or level < 0
                or section_index < 0
            ):
                return ()

            if title is not None and not isinstance(title, str):
                return ()

            if (
                not isinstance(heading_path, list)
                or not all(
                    isinstance(item, str)
                    for item in heading_path
                )
            ):
                return ()

            if not content[start_offset:end_offset].strip():
                return ()

            sections.append(
                _SectionBoundary(
                    section_index=section_index,
                    title=title,
                    level=level,
                    heading_path=tuple(heading_path),
                    start_offset=start_offset,
                    end_offset=end_offset,
                )
            )

        sections.sort(
            key=lambda section: section.start_offset
        )

        cursor = 0

        for section in sections:
            if section.start_offset < cursor:
                return ()

            if content[cursor:section.start_offset].strip():
                return ()

            cursor = section.end_offset

        if content[cursor:].strip():
            return ()

        return tuple(sections)

    @staticmethod
    def _is_integer(value: object) -> bool:
        """排除 bool 后判断整数，避免 JSON true/false 被当作 offset。"""

        return isinstance(value, int) and not isinstance(value, bool)
