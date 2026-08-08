from __future__ import annotations

from collections.abc import Sequence
import json
import math
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.services.reranker.base import RerankItem, RerankerProvider


class BailianRerankerProvider(RerankerProvider):
    """
    阿里云百炼 qwen3-rerank Provider。

    使用 OpenAI-compatible rerank HTTP 接口，不额外引入 SDK 依赖。
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str = "qwen3-rerank",
        timeout: int = 30,
        instruct: str | None = None,
    ) -> None:
        normalized_api_key = api_key.strip()
        normalized_base_url = base_url.strip().rstrip("/")
        normalized_model = model.strip()

        if not normalized_api_key:
            raise ValueError("reranker api_key is required")
        if not normalized_base_url:
            raise ValueError("reranker base_url is required")
        if not normalized_model:
            raise ValueError("reranker model is required")
        if timeout <= 0:
            raise ValueError("reranker timeout must be greater than zero")

        self.api_key = normalized_api_key
        self.base_url = normalized_base_url
        self.model = normalized_model
        self.timeout = timeout
        self.instruct = instruct.strip() if instruct and instruct.strip() else None

    @property
    def model_name(self) -> str:
        """返回当前重排序模型名称。"""

        return self.model

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_n: int,
    ) -> list[RerankItem]:
        """调用百炼 rerank API 对候选文本重新排序。"""

        normalized_query = query.strip()
        normalized_documents = [document.strip() for document in documents]

        if not normalized_query:
            raise ValueError("rerank query cannot be empty")
        if not normalized_documents:
            return []
        if any(not document for document in normalized_documents):
            raise ValueError("rerank documents cannot contain empty text")
        if top_n <= 0:
            raise ValueError("rerank top_n must be greater than zero")

        resolved_top_n = min(top_n, len(normalized_documents))

        body: dict[str, object] = {
            "model": self.model,
            "query": normalized_query,
            "documents": normalized_documents,
            "top_n": resolved_top_n,
        }
        if self.instruct is not None:
            body["instruct"] = self.instruct

        request = Request(
            url=f"{self.base_url}/reranks",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = self._read_http_error(exc)
            raise RuntimeError(
                f"reranker request failed: HTTP {exc.code}: {detail}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(
                f"reranker request failed: {exc.reason}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError("reranker response is not valid JSON") from exc

        return self._parse_results(
            payload=payload,
            document_count=len(normalized_documents),
        )

    @staticmethod
    def _read_http_error(exc: HTTPError) -> str:
        """读取 HTTP 错误体，避免丢失服务端返回的原因。"""

        try:
            content = exc.read().decode("utf-8").strip()
        except Exception:
            return str(exc.reason)
        return content or str(exc.reason)

    @staticmethod
    def _parse_results(
        payload: object,
        document_count: int,
    ) -> list[RerankItem]:
        """校验并转换 qwen3-rerank 返回结果。"""

        if not isinstance(payload, dict):
            raise RuntimeError("reranker response must be an object")

        raw_results = payload.get("results")
        if not isinstance(raw_results, list):
            raise RuntimeError("reranker response missing results")

        items: list[RerankItem] = []
        seen_indexes: set[int] = set()

        for raw_result in raw_results:
            if not isinstance(raw_result, dict):
                raise RuntimeError("reranker result must be an object")

            index = raw_result.get("index")
            score = raw_result.get("relevance_score")

            if isinstance(index, bool) or not isinstance(index, int):
                raise RuntimeError("reranker result index is invalid")
            if not 0 <= index < document_count:
                raise RuntimeError("reranker result index is out of range")
            if index in seen_indexes:
                raise RuntimeError("reranker result index is duplicated")
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise RuntimeError("reranker relevance_score is invalid")

            normalized_score = float(score)
            if not math.isfinite(normalized_score):
                raise RuntimeError("reranker relevance_score is not finite")

            seen_indexes.add(index)
            items.append(
                RerankItem(
                    index=index,
                    score=normalized_score,
                )
            )

        items.sort(key=lambda item: (-item.score, item.index))
        return items
