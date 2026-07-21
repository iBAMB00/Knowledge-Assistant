from app.services.chunking.base import ChunkStrategy
from app.services.chunking.recursive_character import RecursiveCharacterChunkStrategy



class ChunkStrategyFactory:
    """
    Chunk策略工厂。

    根据策略名称创建对应的 ChunkStrategy。

    不负责：
    - 文本切片逻辑
    - Chunk结果处理
    """


    @staticmethod
    def create(
        strategy_name: str,
    ) -> ChunkStrategy:
        """
        创建指定名称的切片策略。

        Args:
            strategy_name:
                策略名称。

        Returns:
            ChunkStrategy实例。
        """

        if strategy_name == "recursive_character":
            return RecursiveCharacterChunkStrategy()


        raise ValueError(
            f"Unknown chunk strategy: {strategy_name}"
        )