from abc import ABC, abstractmethod
from typing import Any

from app.schemas.chunk import ChunkResult


class ChunkStrategy(ABC):
    """
    文本切片策略抽象接口。

    所有切片算法都必须遵守该接口，
    从而避免上层服务依赖某一种具体切片算法。
    """

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """
        返回策略唯一名称。

        示例：
            recursive_character
            markdown
            token
            semantic
        """

        raise NotImplementedError

    @abstractmethod
    def split(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[ChunkResult]:
        """
        将完整文本切分成标准结果。

        Args:
            content:
                文档解析后的完整文本。

            metadata:
                文档级元数据。具体策略可以将其复制或扩展
                到每个 ChunkResult 中。

        Returns:
            按原文顺序排列的 ChunkResult 列表。
        """

        raise NotImplementedError