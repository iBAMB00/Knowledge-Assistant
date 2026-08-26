from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from threading import Lock
from typing import Literal


RerankerTokenCountSource = Literal[
    "provider_usage",
    "local_tokenizer",
    "unavailable",
]


@dataclass(frozen=True)
class RerankItem:
    """单个重排结果，index 对应输入 documents 的位置。"""

    index: int
    score: float


@dataclass(frozen=True)
class RerankerCallUsage:
    """一次 Reranker 调用的可观测资源使用。"""

    candidate_count: int
    total_tokens: int | None = None
    token_count_source: RerankerTokenCountSource = "unavailable"

    def __post_init__(self) -> None:
        if self.candidate_count < 0:
            raise ValueError("candidate_count cannot be negative")
        if self.total_tokens is not None and self.total_tokens < 0:
            raise ValueError("total_tokens cannot be negative")
        if (
            self.token_count_source in {
                "provider_usage",
                "local_tokenizer",
            }
            and self.total_tokens is None
        ):
            raise ValueError(
                "reported token usage requires total_tokens"
            )


@dataclass(frozen=True)
class RerankResponse:
    """一次重排调用的排序结果与调用级 Usage。"""

    items: tuple[RerankItem, ...]
    usage: RerankerCallUsage


@dataclass(frozen=True)
class RerankerUsageSnapshot:
    """一段评估运行期间累计的 Reranker Usage 快照。"""

    request_count: int
    successful_request_count: int
    failed_request_count: int
    candidate_count: int
    provider_usage_request_count: int
    provider_total_tokens: int | None
    reported_token_request_count: int
    reported_total_tokens: int | None
    average_candidates_per_request: float
    average_provider_tokens_per_reported_request: float | None
    p95_provider_tokens_per_reported_request: float | None
    average_reported_tokens_per_request: float | None
    p95_reported_tokens_per_request: float | None
    token_count_source: Literal[
        "provider_usage",
        "local_tokenizer",
        "mixed",
        "unavailable",
        "not_applicable",
    ]
    usage_complete: bool

    @classmethod
    def empty(cls) -> "RerankerUsageSnapshot":
        """返回没有发生 Reranker 调用的空快照。"""

        return cls(
            request_count=0,
            successful_request_count=0,
            failed_request_count=0,
            candidate_count=0,
            provider_usage_request_count=0,
            provider_total_tokens=0,
            reported_token_request_count=0,
            reported_total_tokens=0,
            average_candidates_per_request=0.0,
            average_provider_tokens_per_reported_request=None,
            p95_provider_tokens_per_reported_request=None,
            average_reported_tokens_per_request=None,
            p95_reported_tokens_per_request=None,
            token_count_source="not_applicable",
            usage_complete=True,
        )


class RerankerUsageCollector:
    """
    只累计安全的数值 Usage，不保留 Query 或候选文本。

    该 Collector 由评估运行显式注入 RetrievalService，生产链路不注入时
    不产生额外状态，也不会把原始检索内容持久化到观测对象。
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._request_count = 0
        self._successful_request_count = 0
        self._failed_request_count = 0
        self._candidate_count = 0
        self._reported_token_counts: list[tuple[RerankerTokenCountSource, int]] = []

    def record_success(
        self,
        usage: RerankerCallUsage,
    ) -> None:
        """记录一次成功调用以及可观测 Token Usage。"""

        with self._lock:
            self._request_count += 1
            self._successful_request_count += 1
            self._candidate_count += usage.candidate_count
            if (
                usage.token_count_source != "unavailable"
                and usage.total_tokens is not None
            ):
                self._reported_token_counts.append(
                    (usage.token_count_source, usage.total_tokens)
                )

    def record_failure(
        self,
        candidate_count: int,
    ) -> None:
        """记录一次失败调用；失败请求的 Provider Token 通常不可知。"""

        if candidate_count < 0:
            raise ValueError("candidate_count cannot be negative")

        with self._lock:
            self._request_count += 1
            self._failed_request_count += 1
            self._candidate_count += candidate_count

    def snapshot(self) -> RerankerUsageSnapshot:
        """生成线程安全的冻结累计快照。"""

        with self._lock:
            request_count = self._request_count
            successful_request_count = (
                self._successful_request_count
            )
            failed_request_count = self._failed_request_count
            candidate_count = self._candidate_count
            reported_token_counts = list(self._reported_token_counts)

        provider_counts = [
            count
            for source, count in reported_token_counts
            if source == "provider_usage"
        ]
        all_counts = [count for _, count in reported_token_counts]
        reported_sources = {
            source for source, _ in reported_token_counts
        }

        provider_usage_request_count = len(provider_counts)
        provider_total_tokens: int | None
        if request_count == 0:
            provider_total_tokens = 0
        elif provider_usage_request_count == 0:
            provider_total_tokens = None
        else:
            provider_total_tokens = sum(provider_counts)

        reported_token_request_count = len(all_counts)
        reported_total_tokens: int | None
        if request_count == 0:
            reported_total_tokens = 0
        elif reported_token_request_count == 0:
            reported_total_tokens = None
        else:
            reported_total_tokens = sum(all_counts)

        average_candidates = (
            candidate_count / request_count
            if request_count
            else 0.0
        )
        average_provider_tokens = (
            sum(provider_counts) / provider_usage_request_count
            if provider_usage_request_count
            else None
        )
        p95_provider_tokens = (
            self._percentile(provider_counts, 0.95)
            if provider_counts
            else None
        )
        average_reported_tokens = (
            sum(all_counts) / reported_token_request_count
            if reported_token_request_count
            else None
        )
        p95_reported_tokens = (
            self._percentile(all_counts, 0.95)
            if all_counts
            else None
        )

        if request_count == 0:
            token_count_source = "not_applicable"
        elif not reported_sources:
            token_count_source = "unavailable"
        elif reported_sources == {"provider_usage"}:
            token_count_source = "provider_usage"
        elif reported_sources == {"local_tokenizer"}:
            token_count_source = "local_tokenizer"
        else:
            token_count_source = "mixed"

        usage_complete = (
            failed_request_count == 0
            and reported_token_request_count == request_count
        )

        return RerankerUsageSnapshot(
            request_count=request_count,
            successful_request_count=successful_request_count,
            failed_request_count=failed_request_count,
            candidate_count=candidate_count,
            provider_usage_request_count=(
                provider_usage_request_count
            ),
            provider_total_tokens=provider_total_tokens,
            reported_token_request_count=reported_token_request_count,
            reported_total_tokens=reported_total_tokens,
            average_candidates_per_request=average_candidates,
            average_provider_tokens_per_reported_request=(
                average_provider_tokens
            ),
            p95_provider_tokens_per_reported_request=(
                p95_provider_tokens
            ),
            average_reported_tokens_per_request=(
                average_reported_tokens
            ),
            p95_reported_tokens_per_request=(
                p95_reported_tokens
            ),
            token_count_source=token_count_source,
            usage_complete=usage_complete,
        )

    @staticmethod
    def _percentile(
        values: Sequence[int],
        percentile: float,
    ) -> float:
        if not values:
            raise ValueError("values cannot be empty")
        if not 0.0 <= percentile <= 1.0:
            raise ValueError(
                "percentile must be between 0 and 1"
            )

        sorted_values = sorted(values)
        if len(sorted_values) == 1:
            return float(sorted_values[0])

        position = (len(sorted_values) - 1) * percentile
        lower = int(position)
        upper = min(lower + 1, len(sorted_values) - 1)
        if lower == upper:
            return float(sorted_values[lower])

        fraction = position - lower
        return (
            sorted_values[lower]
            + (
                sorted_values[upper]
                - sorted_values[lower]
            )
            * fraction
        )


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

    def rerank_with_usage(
        self,
        query: str,
        documents: Sequence[str],
        top_n: int,
    ) -> RerankResponse:
        """
        返回排序结果与 Usage。

        默认实现保持现有 Provider / 测试替身兼容：如果 Provider 没有
        暴露真实 Usage，只返回 unavailable。支持 Usage 的 Provider 可覆盖。
        """

        items = self.rerank(
            query=query,
            documents=documents,
            top_n=top_n,
        )
        return RerankResponse(
            items=tuple(items),
            usage=RerankerCallUsage(
                candidate_count=len(documents),
                total_tokens=None,
                token_count_source="unavailable",
            ),
        )
