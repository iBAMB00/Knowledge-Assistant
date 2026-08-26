from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from app.core.config import get_settings
from app.core.database import SessionLocal
from app.repositories.document_chunk_repository import DocumentChunkRepository
from app.repositories.document_content_repository import DocumentContentRepository
from app.repositories.document_repository import DocumentRepository
from app.schemas.retrieval_ablation import (
    RetrievalAblationConfiguration,
    RetrievalAblationVariant,
)
from app.services.bm25_retrieval_service import BM25RetrievalService
from app.services.embedding.factory import EmbeddingFactory
from app.services.evaluation.eval_v2_adapter import EvaluationV2Adapter
from app.services.evaluation.retrieval_ablation_service import (
    RetrievalAblationRunner,
    RetrievalAblationVariantRunner,
    build_default_retrieval_ablation_variants,
)
from app.services.evaluation.retrieval_case_loader import RetrievalCaseLoader
from app.services.evaluation.retrieval_dataset_validator import RetrievalDatasetValidator
from app.services.evaluation.retrieval_evaluator import RetrievalEvaluator
from app.services.reranker.base import RerankerUsageCollector
from app.services.reranker.factory import RerankerFactory
from app.services.retrieval_service import RetrievalService
from app.services.rrf_fusion_service import RRFFusionService
from app.services.vector_store.base import VectorStore
from app.services.vector_store.factory import VectorStoreFactory


DEFAULT_CASES_PATH = Path("evaluation/retrieval_cases_v2.json")
DEFAULT_REPORT_PATH = Path("evaluation/reports/retrieval_ablation_v2.json")
DEFAULT_MARKDOWN_PATH = Path("evaluation/reports/retrieval_ablation_v2.md")
DEFAULT_EVAL_V2_DIR = Path("evaluation/reports/eval_v2/retrieval_ablation")
FULL_VARIANT_ID = "full"


@dataclass(frozen=True)
class SharedAblationDependencies:
    embedding_provider: object
    vector_store: VectorStore
    document_chunk_repository: DocumentChunkRepository
    bm25_retriever: BM25RetrievalService
    rrf_fusion_service: RRFFusionService
    reranker: object


settings = get_settings()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen 90-case Retrieval Eval as a controlled ablation "
            "matrix and emit Eval 2.0 compatible evidence."
        )
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN_PATH)
    parser.add_argument("--eval-v2-dir", type=Path, default=DEFAULT_EVAL_V2_DIR)
    parser.add_argument("--top-k", type=int, default=settings.retrieval_top_k)
    parser.add_argument("--candidate-k", type=int, default=settings.retrieval_candidate_k)
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=settings.retrieval_score_threshold,
    )
    parser.add_argument(
        "--per-document-limit",
        type=int,
        default=settings.retrieval_per_document_limit,
    )
    parser.add_argument("--code-version", type=str, default=None)
    parser.add_argument(
        "--cost-currency",
        type=str,
        default="CNY",
        help="Currency label used for Reranker token cost. Defaults to CNY.",
    )
    parser.add_argument(
        "--reranker-price-per-million-tokens",
        type=float,
        default=0.0,
        help=(
            "Reranker provider price per 1,000,000 reported tokens. "
            "Defaults to 0."
        ),
    )
    parser.add_argument(
        "--skip-parent-child-runtime-ablation",
        action="store_true",
        help=(
            "Skip full-minus-parent-child. This variant is runtime-only on the "
            "existing mixed Parent/Child index and is intentionally labeled weaker."
        ),
    )
    args = parser.parse_args()
    args.cost_currency = args.cost_currency.strip()
    if not args.cost_currency:
        parser.error("--cost-currency cannot be empty")
    if args.reranker_price_per_million_tokens < 0.0:
        parser.error("--reranker-price-per-million-tokens cannot be negative")
    return args


def validate_full_profile() -> None:
    missing: list[str] = []
    if not settings.parent_child_enabled:
        missing.append("PARENT_CHILD_ENABLED=True")
    if not settings.retrieval_hybrid_enabled:
        missing.append("RETRIEVAL_HYBRID_ENABLED=True")
    if not settings.reranker_enabled:
        missing.append("RERANKER_ENABLED=True")
    if missing:
        raise RuntimeError(
            "A7 full ablation profile is incomplete: " + ", ".join(missing)
        )


def build_shared_dependencies() -> SharedAblationDependencies:
    embedding_provider = EmbeddingFactory.create()
    document_chunk_repository = DocumentChunkRepository()
    vector_store = VectorStoreFactory.create(
        settings=settings,
    ).vector_store
    return SharedAblationDependencies(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        document_chunk_repository=document_chunk_repository,
        bm25_retriever=BM25RetrievalService(
            document_chunk_repository=document_chunk_repository,
        ),
        rrf_fusion_service=RRFFusionService(
            rank_constant=settings.retrieval_rrf_k,
        ),
        reranker=RerankerFactory.create(),
    )


def build_evaluator(
    shared: SharedAblationDependencies,
    variant: RetrievalAblationVariant,
    reranker_usage_collector: RerankerUsageCollector | None = None,
) -> RetrievalEvaluator:
    service = RetrievalService(
        embedding_provider=shared.embedding_provider,
        vector_store=shared.vector_store,
        default_top_k=settings.retrieval_top_k,
        default_candidate_k=settings.retrieval_candidate_k,
        default_score_threshold=settings.retrieval_score_threshold,
        default_per_document_limit=settings.retrieval_per_document_limit,
        document_chunk_repository=shared.document_chunk_repository,
        parent_child_enabled=variant.parent_child_enabled,
        bm25_retriever=shared.bm25_retriever,
        rrf_fusion_service=shared.rrf_fusion_service,
        hybrid_enabled=variant.hybrid_rrf_enabled,
        reranker=shared.reranker,
        reranker_enabled=variant.reranker_enabled,
        reranker_fail_open=settings.reranker_fail_open,
        reranker_usage_collector=reranker_usage_collector,
    )
    return RetrievalEvaluator(retrieval_service=service)


def build_variant_runner(
    shared: SharedAblationDependencies,
    variant: RetrievalAblationVariant,
) -> RetrievalAblationVariantRunner:
    """为每个 Variant 创建独立 Usage Collector，避免跨变体串账。"""

    collector = RerankerUsageCollector()
    return RetrievalAblationVariantRunner(
        variant=variant,
        evaluator=build_evaluator(
            shared=shared,
            variant=variant,
            reranker_usage_collector=collector,
        ),
        reranker_usage_collector=collector,
    )


def build_validator() -> RetrievalDatasetValidator:
    return RetrievalDatasetValidator(
        document_repository=DocumentRepository(),
        document_content_repository=DocumentContentRepository(),
        document_chunk_repository=DocumentChunkRepository(),
    )


def resolve_code_version() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    value = completed.stdout.strip()
    return value or None


def build_configuration(
    args: argparse.Namespace,
    shared: SharedAblationDependencies,
    code_version: str | None,
) -> RetrievalAblationConfiguration:
    """生成不包含密钥的消融实验运行配置快照。"""

    return RetrievalAblationConfiguration(
        code_version=code_version,
        vector_store_backend=settings.vector_store_backend,
        embedding_provider=settings.embedding_provider,
        embedding_model=shared.embedding_provider.model_name,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        score_threshold=args.score_threshold,
        per_document_limit=args.per_document_limit,
        shared_query_embedding=True,
        reranker_model=settings.reranker_model,
        reranker_fail_open=settings.reranker_fail_open,
        cost_currency=getattr(args, "cost_currency", "CNY"),
        reranker_price_per_million_tokens=getattr(
            args,
            "reranker_price_per_million_tokens",
            0.0,
        ),
    )


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_markdown(path: Path, report) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Retrieval Ablation v2",
        "",
        f"Dataset: `{report.dataset.dataset_id}@{report.dataset.dataset_version}`",
        "",
        f"Vector store: `{report.configuration.vector_store_backend}`",
        "",
        f"Reranker: `{report.configuration.reranker_model}`",
        "",
        (
            "Shared query embedding tokens: "
            f"`{report.shared_token_usage.total_query_embedding_tokens}` "
            f"across `{report.shared_token_usage.request_count}` requests "
            f"({report.shared_token_usage.token_count_source})"
        ),
        "",
        "| Variant | Evidence | Chunk Recall | Chunk MRR | Chunk nDCG | Context Avg | Rerank Req | Rerank Candidates | Reranker Tokens | Token Source | Usage Complete | Retrieval ms | P95 Total ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|",
    ]
    for result in report.variants:
        m = result.metrics
        context = "n/a" if m.average_context_tokens is None else f"{m.average_context_tokens:.2f}"
        reranker_usage = result.token_usage.reranker
        reranker_tokens = (
            "n/a"
            if reranker_usage.reported_total_tokens is None
            else str(reranker_usage.reported_total_tokens)
        )
        lines.append(
            "| "
            f"{result.variant.variant_id} | {result.variant.evidence_strength.value} | "
            f"{m.chunk_recall_at_k:.4f} | {m.chunk_mrr:.4f} | "
            f"{m.chunk_ndcg_at_k:.4f} | {context} | "
            f"{reranker_usage.request_count} | {reranker_usage.candidate_count} | "
            f"{reranker_tokens} | {reranker_usage.token_count_source} | "
            f"{reranker_usage.usage_complete} | "
            f"{m.average_retrieval_latency_ms:.2f} | {m.p95_total_latency_ms:.2f} |"
        )
    lines.extend(["", "## Full-vs-Variant Deltas", ""])
    lines.extend(
        [
            "| Variant | Removed component | Evidence | Metric | Full | Variant | Full advantage |",
            "|---|---|---|---|---:|---:|---:|",
        ]
    )
    key_metrics = {
        "chunk_recall_at_k",
        "chunk_mrr",
        "chunk_ndcg_at_k",
        "average_context_tokens",
        "average_retrieval_latency_ms",
        "p95_total_latency_ms",
    }
    for delta in report.deltas:
        if delta.removed_component is None or delta.metric_id not in key_metrics:
            continue
        lines.append(
            "| "
            f"{delta.variant_id} | {delta.removed_component.value} | "
            f"{delta.evidence_strength.value} | {delta.metric_id} | "
            f"{delta.full_value:.4f} | {delta.variant_value:.4f} | "
            f"{delta.full_advantage:+.4f} |"
        )

    lines.extend(["", "## Token Attribution", ""])
    lines.append(
        f"- Shared Query Embedding: {report.shared_token_usage.total_query_embedding_tokens} "
        f"tokens / {report.shared_token_usage.request_count} requests "
        f"({report.shared_token_usage.token_count_source})."
    )
    for result in report.variants:
        usage = result.token_usage.reranker
        reranker_tokens = (
            "n/a"
            if usage.reported_total_tokens is None
            else str(usage.reported_total_tokens)
        )
        context_usage = result.token_usage.final_context
        context_total = (
            "n/a"
            if context_usage is None
            else str(context_usage.total_context_tokens)
        )
        lines.append(
            f"- `{result.variant.variant_id}`: reranker requests={usage.request_count}, "
            f"candidates={usage.candidate_count}, reranker_tokens={reranker_tokens}, "
            f"source={usage.token_count_source}, complete={usage.usage_complete}, "
            f"final_context_tokens={context_total}."
        )

    lines.extend(["", "## Regression Cases", ""])
    regressions_by_variant: dict[str, list] = {}
    for regression in report.case_regressions:
        regressions_by_variant.setdefault(regression.variant_id, []).append(regression)
    if not regressions_by_variant:
        lines.append("No case-level quality regressions detected relative to Full.")
    else:
        for variant_id, regressions in sorted(regressions_by_variant.items()):
            case_ids = ", ".join(item.case_id for item in regressions[:10])
            suffix = "" if len(regressions) <= 10 else f" ... (+{len(regressions) - 10})"
            lines.append(
                f"- `{variant_id}`: {len(regressions)} regression case(s): "
                f"{case_ids}{suffix}"
            )

    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in report.limitations)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    validate_full_profile()

    dataset = RetrievalCaseLoader.load(args.cases)
    dataset_reference = RetrievalCaseLoader.build_reference(dataset, args.cases)
    shared = build_shared_dependencies()
    variant_specs = build_default_retrieval_ablation_variants(
        include_parent_child_runtime_ablation=(
            not args.skip_parent_child_runtime_ablation
        ),
    )
    runners = tuple(
        build_variant_runner(shared, variant)
        for variant in variant_specs
    )
    code_version = args.code_version or resolve_code_version()
    configuration = build_configuration(
        args=args,
        shared=shared,
        code_version=code_version,
    )

    with SessionLocal() as db:
        build_validator().validate(db, dataset)
        report = RetrievalAblationRunner(
            variants=runners,
            full_variant_id=FULL_VARIANT_ID,
            document_chunk_repository=shared.document_chunk_repository,
        ).run(
            db=db,
            cases=dataset.cases,
            dataset=dataset_reference,
            configuration=configuration,
        )

    write_json(args.output, report)
    write_markdown(args.markdown_output, report)

    args.eval_v2_dir.mkdir(parents=True, exist_ok=True)
    for result in report.variants:
        context_usage = result.token_usage.final_context
        reranker_usage = result.token_usage.reranker
        eval_run = EvaluationV2Adapter.from_retrieval_run(
            dataset=dataset_reference,
            run=result.run,
            generated_at=report.generated_at,
            retrieval_variant_id=result.variant.variant_id,
            code_version=code_version,
            average_context_tokens=(
                context_usage.average_context_tokens
                if context_usage is not None
                else None
            ),
            reranker_tokens=(
                reranker_usage.reported_total_tokens
            ),
            reranker_request_count=reranker_usage.request_count,
        )
        write_json(args.eval_v2_dir / f"{result.variant.variant_id}.json", eval_run)

    print(
        json.dumps(
            {
                "decision": "completed",
                "dataset": f"{dataset.dataset_id}@{dataset.dataset_version}",
                "cases": len(dataset.cases),
                "variants": [variant.variant_id for variant in variant_specs],
                "shared_query_embedding": True,
                "shared_query_embedding_tokens": (
                    report.shared_token_usage.total_query_embedding_tokens
                ),
                "reranker_tokens_by_variant": {
                    result.variant.variant_id: (
                        result.token_usage.reranker.reported_total_tokens
                    )
                    for result in report.variants
                },
                "report": str(args.output),
                "markdown": str(args.markdown_output),
                "eval_v2_dir": str(args.eval_v2_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
