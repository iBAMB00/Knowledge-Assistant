from abc import ABC, abstractmethod
from collections.abc import Sequence


class EmbeddingProvider(ABC):
    """
    Embedding模型抽象接口。

    文档切片向量和查询向量使用独立方法，
    便于未来接入需要区分query和document前缀的模型。
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """
        返回当前Embedding模型名称。
        """

        raise NotImplementedError

    @abstractmethod
    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        """
        批量生成文档切片向量。

        返回顺序必须与输入文本顺序一致。
        """

        raise NotImplementedError

    @abstractmethod
    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        """
        生成查询文本向量。
        """

        raise NotImplementedError