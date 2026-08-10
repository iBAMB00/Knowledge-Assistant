from typing import Any

from app.schemas.chunk import ChunkResult
from app.services.chunking.factory import (
    ChunkStrategyFactory,
)
from app.services.chunking.section_aware import (
    SectionAwareParentChunker,
)


class ChunkService:
    """
    文档切片服务。

    负责调用切片策略完成文本切分。

    不负责：
    - 切片算法
    - 数据库存储
    """


    def split(
        self,
        content: str,
        strategy_name: str,
        metadata: dict[str, Any] | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> list[ChunkResult]:
        """
        使用指定策略切分文本。
        """

        strategy = ChunkStrategyFactory.create(
            strategy_name=strategy_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        return strategy.split(
            content=content,
            metadata=metadata,
        )

    def split_parent_by_structure(
        self,
        content: str,
        strategy_name: str,
        structure_metadata: dict[str, Any] | None,
        metadata: dict[str, Any] | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> list[ChunkResult]:
        """
        优先按章节边界生成 Parent Chunk。

        章节过长时继续复用指定基础策略；结构无效时返回空列表，
        由编排层决定是否降级到普通全文切片。
        """

        strategy = ChunkStrategyFactory.create(
            strategy_name=strategy_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        return SectionAwareParentChunker(
            base_strategy=strategy,
        ).split(
            content=content,
            structure_metadata=structure_metadata,
            metadata=metadata,
        )

