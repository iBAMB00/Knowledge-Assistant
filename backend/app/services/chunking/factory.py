from app.core.config import get_settings
from app.services.chunking.base import ChunkStrategy
from app.services.chunking.recursive_character import (
    RECURSIVE_CHARACTER_STRATEGY_NAME,
    RecursiveCharacterChunkStrategy,
)


class ChunkStrategyFactory:
    """
    Chunk切片策略工厂。

    根据策略名称和应用配置创建具体切片策略。
    """

    @staticmethod
    def create(
        strategy_name: str,
    ) -> ChunkStrategy:
        """
        创建指定的Chunk切片策略。
        """

        normalized_strategy_name = (
            strategy_name.strip().lower()
        )

        settings = get_settings()

        if (
            normalized_strategy_name
            == RECURSIVE_CHARACTER_STRATEGY_NAME
        ):
            return RecursiveCharacterChunkStrategy(
                chunk_size=settings.chunk_size,
                chunk_overlap=(
                    settings.chunk_overlap
                ),
            )

        raise ValueError(
            "unsupported chunk strategy: "
            f"{normalized_strategy_name}"
        )