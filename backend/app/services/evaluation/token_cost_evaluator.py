from collections.abc import Sequence
from dataclasses import dataclass
import math
import re

from sqlalchemy.orm import Session

from app.repositories.document_chunk_repository import DocumentChunkRepository
from app.repositories.document_content_repository import DocumentContentRepository
from app.schemas.retrieval_evaluation import (
    RetrievalEvaluationDatasetReference,
    RetrievalEvaluationRun,
    RetrievalTokenCostCaseUsage,
    RetrievalTokenCostIngestion,
    RetrievalTokenCostModeUsage,
    RetrievalTokenCostPricing,
    RetrievalTokenCostReport,
)


class LocalEstimatedTokenCounter:
    """无外部Tokenizer依赖的本地Token估算器。"""

    name = "unicode_heuristic_v1"
    source = "local_estimation"
    _ASCII_WORD_PATTERN = re.compile(r"[A-Za-z0-9_]+")

    def count(self, text: str) -> int:
        """按中文字符、ASCII词片段和符号估算Token数量。"""
        if not text:
            return 0

        total = 0
        index = 0
        while index < len(text):
            char = text[index]
            if char.isspace():
                index += 1
                continue
            if self._is_cjk_like(char):
                total += 1
                index += 1
                continue

            match = self._ASCII_WORD_PATTERN.match(text, index)
            if match:
                total += max(1, math.ceil(len(match.group(0)) / 4))
                index = match.end()
                continue

            total += 1
            index += 1

        return total

    @staticmethod
    def _is_cjk_like(char: str) -> bool:
        codepoint = ord(char)
        return (
            0x3400 <= codepoint <= 0x9FFF
            or 0x3040 <= codepoint <= 0x30FF
            or 0xAC00 <= codepoint <= 0xD7AF
        )


@dataclass(frozen=True)
class TokenCostEvaluationOptions:
    """Token成本评估参数。"""

    currency: str = "CNY"
    embedding_price_per_million_tokens: float = 0.0
    llm_input_price_per_million_tokens: float = 0.0


class RetrievalTokenCostEvaluator:
    """统计检索评估的入库、查询和召回上下文Token成本。"""

    def __init__(
        self,
        document_content_repository: DocumentContentRepository,
        document_chunk_repository: DocumentChunkRepository,
        token_counter: LocalEstimatedTokenCounter | None = None,
    ) -> None:
        self.document_content_repository = document_content_repository
        self.document_chunk_repository = document_chunk_repository
        self.token_counter = token_counter or LocalEstimatedTokenCounter()

    def evaluate(
        self,
        db: Session,
        dataset: RetrievalEvaluationDatasetReference,
        baseline: RetrievalEvaluationRun,
        optimized: RetrievalEvaluationRun,
        options: TokenCostEvaluationOptions,
    ) -> RetrievalTokenCostReport:
        """生成检索阶段Token与成本报告。"""
        pricing = RetrievalTokenCostPricing(
            currency=options.currency,
            embedding_price_per_million_tokens=options.embedding_price_per_million_tokens,
            llm_input_price_per_million_tokens=options.llm_input_price_per_million_tokens,
        )
        ingestion = self._build_ingestion_usage(db, dataset, pricing)
        all_chunk_ids = {
            chunk_id
            for run in (baseline, optimized)
            for case in run.cases
            for chunk_id in case.retrieved_chunk_ids
        }
        chunk_map = {chunk.id: chunk for chunk in self.document_chunk_repository.find_by_ids(db, sorted(all_chunk_ids))}
        query_tokens = {case.case_id: self.token_counter.count(case.question) for case in baseline.cases}
        baseline_context = self._build_context_tokens(baseline, chunk_map)
        optimized_context = self._build_context_tokens(optimized, chunk_map)
        cases = [
            RetrievalTokenCostCaseUsage(
                case_id=case.case_id,
                query_tokens=query_tokens[case.case_id],
                baseline_context_tokens=baseline_context[case.case_id],
                optimized_context_tokens=optimized_context[case.case_id],
            )
            for case in baseline.cases
        ]
        total_query_tokens = sum(query_tokens.values())

        return RetrievalTokenCostReport(
            token_count_source=self.token_counter.source,
            tokenizer_name=self.token_counter.name,
            pricing=pricing,
            ingestion=ingestion,
            total_query_embedding_tokens=total_query_tokens,
            average_query_embedding_tokens=self._mean(list(query_tokens.values())),
            p50_query_embedding_tokens=self._percentile(list(query_tokens.values()), 0.50),
            p95_query_embedding_tokens=self._percentile(list(query_tokens.values()), 0.95),
            estimated_query_embedding_cost=self._cost(total_query_tokens, pricing.embedding_price_per_million_tokens),
            baseline=self._build_mode_usage(baseline_context, total_query_tokens, pricing),
            optimized=self._build_mode_usage(optimized_context, total_query_tokens, pricing),
            cases=cases,
        )

    def _build_ingestion_usage(
        self,
        db: Session,
        dataset: RetrievalEvaluationDatasetReference,
        pricing: RetrievalTokenCostPricing,
    ) -> RetrievalTokenCostIngestion:
        contents = self.document_content_repository.find_by_document_ids(db, dataset.corpus_document_ids)
        if len(contents) != len(dataset.corpus_document_ids):
            missing = sorted(set(dataset.corpus_document_ids) - set(contents))
            raise ValueError(f"missing document contents for token evaluation: {missing}")

        content_ids = [content.id for content in contents.values()]
        chunks = self.document_chunk_repository.find_by_document_content_ids(db, content_ids)
        source_tokens = sum(self.token_counter.count(content.content) for content in contents.values())
        chunk_token_counts = [self.token_counter.count(chunk.content) for chunk in chunks]
        embedded_tokens = sum(chunk_token_counts)
        estimated_overlap_extra_tokens = max(0, embedded_tokens - source_tokens)
        overhead_rate = estimated_overlap_extra_tokens / source_tokens if source_tokens else 0.0

        return RetrievalTokenCostIngestion(
            document_count=len(contents),
            chunk_count=len(chunks),
            source_tokens=source_tokens,
            chunk_embedding_tokens=embedded_tokens,
            estimated_overlap_extra_tokens=estimated_overlap_extra_tokens,
            estimated_overlap_overhead_rate=overhead_rate,
            average_chunk_tokens=self._mean(chunk_token_counts),
            p50_chunk_tokens=self._percentile(chunk_token_counts, 0.50),
            p95_chunk_tokens=self._percentile(chunk_token_counts, 0.95),
            estimated_embedding_cost=self._cost(embedded_tokens, pricing.embedding_price_per_million_tokens),
        )

    def _build_context_tokens(self, run: RetrievalEvaluationRun, chunk_map: dict[int, object]) -> dict[str, int]:
        usage: dict[str, int] = {}
        for case in run.cases:
            missing = [chunk_id for chunk_id in case.retrieved_chunk_ids if chunk_id not in chunk_map]
            if missing:
                raise ValueError(f"retrieved chunks missing during token evaluation: {missing}")
            usage[case.case_id] = sum(self.token_counter.count(chunk_map[chunk_id].content) for chunk_id in case.retrieved_chunk_ids)
        return usage

    def _build_mode_usage(
        self,
        context_tokens_by_case: dict[str, int],
        total_query_tokens: int,
        pricing: RetrievalTokenCostPricing,
    ) -> RetrievalTokenCostModeUsage:
        context_values = list(context_tokens_by_case.values())
        case_count = len(context_values)
        total_context_tokens = sum(context_values)
        query_cost = self._cost(total_query_tokens, pricing.embedding_price_per_million_tokens)
        context_cost = self._cost(total_context_tokens, pricing.llm_input_price_per_million_tokens)
        total_cost = query_cost + context_cost
        average_cost = total_cost / case_count if case_count else 0.0

        return RetrievalTokenCostModeUsage(
            case_count=case_count,
            total_context_tokens=total_context_tokens,
            average_context_tokens=self._mean(context_values),
            p50_context_tokens=self._percentile(context_values, 0.50),
            p95_context_tokens=self._percentile(context_values, 0.95),
            estimated_context_input_cost=context_cost,
            estimated_retrieval_stage_cost=total_cost,
            estimated_average_cost_per_query=average_cost,
            estimated_cost_per_1000_queries=average_cost * 1000,
            estimated_cost_per_10000_queries=average_cost * 10000,
        )

    @staticmethod
    def _cost(tokens: int, price_per_million_tokens: float) -> float:
        return tokens / 1_000_000 * price_per_million_tokens

    @staticmethod
    def _mean(values: Sequence[int]) -> float:
        return sum(values) / len(values) if values else 0.0

    @staticmethod
    def _percentile(values: Sequence[int], percentile: float) -> float:
        if not values:
            return 0.0
        sorted_values = sorted(values)
        if len(sorted_values) == 1:
            return float(sorted_values[0])
        position = (len(sorted_values) - 1) * percentile
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return float(sorted_values[lower])
        fraction = position - lower
        return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction
