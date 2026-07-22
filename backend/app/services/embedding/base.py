from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """
    Embedding模型抽象接口。

    具体实现可以接入：
    - OpenAI兼容Embedding接口
    - BGE系列模型
    - 本地Embedding模型
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
        texts: list[str],
    ) -> list[list[float]]:
        """
        批量生成文档切片向量。

        返回结果顺序必须与输入文本顺序一致。
        """

        raise NotImplementedError

    @abstractmethod
    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        """
        生成用户查询文本向量。
        """

        raise NotImplementedError