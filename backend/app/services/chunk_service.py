from typing import Any

from app.schemas.chunk import ChunkResult
from app.services.chunking.factory import (
    ChunkStrategyFactory,
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