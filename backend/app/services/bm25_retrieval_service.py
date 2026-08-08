from __future__ import annotations

from collections import Counter
import math
import re

from sqlalchemy.orm import Session

from app.repositories.document_chunk_repository import DocumentChunkRepository
from app.schemas.vector_search_result import VectorSearchResult
from app.services.vector_store.base import ChunkRole


class BM25RetrievalService:
    """
    轻量 BM25 关键词检索服务。

    当前版本直接从 PostgreSQL 读取可检索 Chunk，适合 MVP 与中小规模
    知识库。中文采用连续汉字 bigram，英文与数字按词切分，避免为
    v0.14 引入额外搜索引擎依赖。
    """

    _TOKEN_PATTERN = re.compile(
        r"[a-zA-Z0-9_./:-]+|[\u4e00-\u9fff]+"
    )

    def __init__(
        self,
        document_chunk_repository: DocumentChunkRepository,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be greater than zero")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")

        self.document_chunk_repository = document_chunk_repository
        self.k1 = float(k1)
        self.b = float(b)

    def search(
        self,
        db: Session,
        query: str,
        top_k: int = 20,
        document_id: int | None = None,
        chunk_role: ChunkRole | None = None,
    ) -> list[VectorSearchResult]:
        """根据 BM25 分数返回关键词相关 Chunk。"""

        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if document_id is not None and document_id <= 0:
            raise ValueError("document_id must be greater than zero")

        query_tokens = self._tokenize(normalized_query)
        if not query_tokens:
            return []

        rows = self.document_chunk_repository.find_retrieval_candidates(
            db=db,
            document_id=document_id,
            chunk_role=chunk_role,
        )
        if not rows:
            return []

        tokenized_documents = [
            self._tokenize(chunk.content)
            for chunk, _, _ in rows
        ]
        average_length = sum(
            len(tokens) for tokens in tokenized_documents
        ) / len(tokenized_documents)
        if average_length <= 0:
            return []

        document_frequency: Counter[str] = Counter()
        for tokens in tokenized_documents:
            document_frequency.update(set(tokens))

        total_documents = len(rows)
        query_terms = set(query_tokens)
        scored_results: list[VectorSearchResult] = []

        for (chunk, document_content, document), tokens in zip(
            rows,
            tokenized_documents,
            strict=True,
        ):
            if not tokens:
                continue

            frequencies = Counter(tokens)
            document_length = len(tokens)
            score = 0.0

            for term in query_terms:
                term_frequency = frequencies.get(term, 0)
                if term_frequency <= 0:
                    continue

                df = document_frequency.get(term, 0)
                idf = math.log(
                    1.0
                    + (
                        total_documents - df + 0.5
                    ) / (df + 0.5)
                )
                denominator = (
                    term_frequency
                    + self.k1
                    * (
                        1.0
                        - self.b
                        + self.b
                        * document_length
                        / average_length
                    )
                )
                score += (
                    idf
                    * term_frequency
                    * (self.k1 + 1.0)
                    / denominator
                )

            if score <= 0:
                continue

            scored_results.append(
                VectorSearchResult(
                    document_id=document_content.document_id,
                    filename=document.filename,
                    chunk_id=chunk.id,
                    chunk_index=chunk.chunk_index,
                    content=chunk.content,
                    score=score,
                    parent_chunk_id=chunk.parent_chunk_id,
                )
            )

        scored_results.sort(
            key=lambda result: (-result.score, result.chunk_id)
        )
        return scored_results[:top_k]

    @classmethod
    def _tokenize(cls, text: str) -> list[str]:
        """将中英文混合文本转换为 BM25 Token。"""

        tokens: list[str] = []

        for match in cls._TOKEN_PATTERN.finditer(text.lower()):
            value = match.group(0)
            if not value:
                continue

            if cls._is_chinese(value):
                if len(value) == 1:
                    tokens.append(value)
                    continue

                tokens.extend(
                    value[index:index + 2]
                    for index in range(len(value) - 1)
                )
                continue

            tokens.append(value)

        return tokens

    @staticmethod
    def _is_chinese(value: str) -> bool:
        return all("\u4e00" <= char <= "\u9fff" for char in value)
