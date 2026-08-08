from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class RerankItem:
    """单个重排结果，index 对应输入 documents 的位置。"""

    index: int
    score: float


class RerankerProvider(ABC):
    """重排序模型抽象。"""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """返回当前重排序模型名称。"""

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_n: int,
    ) -> list[RerankItem]:
        """按与 query 的相关性重新排序候选文档。"""
