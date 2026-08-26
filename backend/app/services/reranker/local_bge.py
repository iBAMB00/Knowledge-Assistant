from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import Any

from app.services.reranker.base import (
    RerankItem,
    RerankerCallUsage,
    RerankerProvider,
    RerankResponse,
)


@lru_cache(maxsize=4)
def _load_local_bge_runtime(
    model_name: str,
    device: str,
) -> tuple[Any, Any, Any]:
    """
    按进程缓存本地 Reranker runtime。

    多个 API / Eval 入口可能分别通过 Factory 创建 Provider；缓存确保同一
    model + device 只加载一份 tokenizer / model，避免重复占用内存。
    """

    try:
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )
    except ImportError as exc:  # pragma: no cover - 由部署环境决定
        raise RuntimeError(
            "local_bge reranker requires optional dependencies: "
            "torch and transformers"
        ) from exc

    resolved_device = torch.device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name
    )
    model.eval()
    model.to(resolved_device)
    return torch, tokenizer, model


class LocalBGERerankerProvider(RerankerProvider):
    """使用本地 Hugging Face Sequence Classification 模型执行重排序。"""

    def __init__(
        self,
        *,
        model: str = "BAAI/bge-reranker-v2-m3",
        device: str = "cpu",
        batch_size: int = 8,
        max_length: int = 512,
    ) -> None:
        normalized_model = model.strip()
        normalized_device = device.strip()

        if not normalized_model:
            raise ValueError("reranker model is required")
        if not normalized_device:
            raise ValueError("reranker local device is required")
        if batch_size <= 0:
            raise ValueError("reranker local batch_size must be greater than zero")
        if max_length <= 0:
            raise ValueError("reranker local max_length must be greater than zero")

        self.model = normalized_model
        self.device = normalized_device
        self.batch_size = batch_size
        self.max_length = max_length
        self._torch, self._tokenizer, self._model = _load_local_bge_runtime(
            self.model,
            self.device,
        )

    @property
    def model_name(self) -> str:
        """返回本地重排序模型名称。"""

        return self.model

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_n: int,
    ) -> list[RerankItem]:
        """保持现有 RerankerProvider 接口兼容。"""

        return list(
            self.rerank_with_usage(
                query=query,
                documents=documents,
                top_n=top_n,
            ).items
        )

    def rerank_with_usage(
        self,
        query: str,
        documents: Sequence[str],
        top_n: int,
    ) -> RerankResponse:
        """
        批量执行 Query-Passage 打分，并使用实际 tokenizer 统计输入 Token。

        本地模型没有云 Provider 账单 Usage，因此 token_count_source 明确标记为
        local_tokenizer；该 Token 用于资源归因，不代表 API 费用。
        """

        normalized_query = query.strip()
        normalized_documents = [document.strip() for document in documents]

        if not normalized_query:
            raise ValueError("rerank query cannot be empty")
        if not normalized_documents:
            return RerankResponse(
                items=(),
                usage=RerankerCallUsage(
                    candidate_count=0,
                    total_tokens=0,
                    token_count_source="local_tokenizer",
                ),
            )
        if any(not document for document in normalized_documents):
            raise ValueError("rerank documents cannot contain empty text")
        if top_n <= 0:
            raise ValueError("rerank top_n must be greater than zero")

        resolved_top_n = min(top_n, len(normalized_documents))
        scored_items: list[RerankItem] = []
        total_tokens = 0

        for start in range(0, len(normalized_documents), self.batch_size):
            batch_documents = normalized_documents[
                start : start + self.batch_size
            ]
            pairs = [
                [normalized_query, document]
                for document in batch_documents
            ]
            encoded = self._tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.max_length,
            )

            attention_mask = encoded.get("attention_mask")
            if attention_mask is None:
                raise RuntimeError(
                    "local_bge tokenizer response missing attention_mask"
                )
            total_tokens += int(attention_mask.sum().item())

            model_inputs = {
                key: value.to(self.device)
                for key, value in encoded.items()
            }
            with self._torch.no_grad():
                output = self._model(
                    **model_inputs,
                    return_dict=True,
                )

            raw_scores = (
                output.logits.view(-1)
                .detach()
                .float()
                .cpu()
                .tolist()
            )
            if len(raw_scores) != len(batch_documents):
                raise RuntimeError(
                    "local_bge model returned unexpected score count"
                )

            scored_items.extend(
                RerankItem(
                    index=start + offset,
                    score=float(score),
                )
                for offset, score in enumerate(raw_scores)
            )

        scored_items.sort(
            key=lambda item: (-item.score, item.index)
        )
        selected = tuple(scored_items[:resolved_top_n])

        return RerankResponse(
            items=selected,
            usage=RerankerCallUsage(
                candidate_count=len(normalized_documents),
                total_tokens=total_tokens,
                token_count_source="local_tokenizer",
            ),
        )
