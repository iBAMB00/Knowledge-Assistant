import argparse
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import get_settings
from app.core.database import SessionLocal
from app.repositories.chunk_embedding_repository import (
    ChunkEmbeddingRepository,
)
from app.repositories.document_chunk_repository import (
    DocumentChunkRepository,
)
from app.repositories.document_content_repository import (
    DocumentContentRepository,
)
from app.repositories.document_repository import (
    DocumentRepository,
)
from app.schemas.retrieval_evaluation import (
    RetrievalComparisonReport,
    RetrievalEvaluationConfiguration,
    RetrievalEvaluationSummary,
    RetrievalRegressionGateThresholds,
)
from app.services.bm25_retrieval_service import (
    BM25RetrievalService,
)
from app.services.embedding.factory import (
    EmbeddingFactory,
)
from app.services.evaluation.retrieval_case_loader import (
    RetrievalCaseLoader,
)
from app.services.evaluation.retrieval_dataset_validator import (
    RetrievalDatasetValidator,
)
from app.services.evaluation.retrieval_evaluator import (
    RetrievalEvaluator,
)
from app.services.evaluation.token_cost_evaluator import (
    RetrievalTokenCostEvaluator,
    TokenCostEvaluationOptions,
)
from app.services.retrieval_service import (
    RetrievalService,
)
from app.services.reranker.factory import (
    RerankerFactory,
)
from app.services.rrf_fusion_service import (
    RRFFusionService,
)
from app.services.vector_store.database import (
    DatabaseVectorStore,
)


DEFAULT_CASES_PATH = Path(
    "evaluation/retrieval_cases_v2.json"
)

DEFAULT_REPORT_PATH = Path(
    "evaluation/reports/"
    "retrieval_comparison_v2.json"
)

settings = get_settings()


@dataclass(frozen=True)
class RetrievalEvaluationComponents:
    """真实检索评估运行组件。"""

    evaluator: RetrievalEvaluator
    embedding_model: str


def parse_args() -> argparse.Namespace:
    """解析检索评估命令行参数。"""

    parser = argparse.ArgumentParser(
        description=(
            "Validate a versioned retrieval dataset "
            "and compare baseline with optimized "
            "retrieval strategies."
        )
    )

    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_CASES_PATH,
        help=(
            "Path to the versioned retrieval "
            "evaluation dataset JSON file."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help=(
            "Path used to save the evaluation "
            "comparison report."
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=settings.retrieval_top_k,
        help="Final number of retrieval results.",
    )

    parser.add_argument(
        "--candidate-k",
        type=int,
        default=settings.retrieval_candidate_k,
        help=(
            "Candidate count used by optimized "
            "retrieval."
        ),
    )

    parser.add_argument(
        "--score-threshold",
        type=float,
        default=settings.retrieval_score_threshold,
        help="Minimum Dense similarity score used before optimized fusion.",
    )

    parser.add_argument(
        "--per-document-limit",
        type=int,
        default=settings.retrieval_per_document_limit,
        help=(
            "Preferred result limit for each "
            "document in optimized retrieval."
        ),
    )

    parser.add_argument(
        "--code-version",
        type=str,
        default=None,
        help=(
            "Code revision written into the report. "
            "Defaults to the current Git commit."
        ),
    )

    parser.add_argument(
        "--quality-regression-tolerance",
        type=float,
        default=0.0,
        help=(
            "Maximum absolute drop allowed for quality "
            "metrics when optimized is compared with "
            "baseline. Defaults to 0."
        ),
    )

    parser.add_argument(
        "--duplicate-rate-regression-tolerance",
        type=float,
        default=0.0,
        help=(
            "Maximum absolute increase allowed for the "
            "duplicate rate. Defaults to 0."
        ),
    )

    parser.add_argument(
        "--latency-regression-ratio",
        type=float,
        default=0.20,
        help=(
            "Maximum relative increase allowed for "
            "average retrieval latency and P95 total "
            "latency. Defaults to 0.20 (20%%)."
        ),
    )

    parser.add_argument(
        "--cost-currency",
        type=str,
        default="CNY",
        help="Currency label used by token cost estimates. Defaults to CNY.",
    )
    parser.add_argument(
        "--embedding-price-per-million-tokens",
        type=float,
        default=0.0,
        help="Embedding price per 1,000,000 tokens. Defaults to 0.",
    )
    parser.add_argument(
        "--llm-input-price-per-million-tokens",
        type=float,
        default=0.0,
        help="LLM input price per 1,000,000 tokens for retrieved context estimation. Defaults to 0.",
    )

    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help=(
            "Exit with status 2 after saving the report "
            "when the regression gate fails."
        ),
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Validate dataset structure and database "
            "references without calling Embedding or "
            "executing retrieval."
        ),
    )

    parser.add_argument(
        "--require-v014-candidate",
        action="store_true",
        help=(
            "Fail fast unless Parent-Child, Hybrid and "
            "Reranker are all enabled for the optimized "
            "v0.14 candidate."
        ),
    )

    args = parser.parse_args()

    if not 0.0 <= args.quality_regression_tolerance <= 1.0:
        parser.error(
            "--quality-regression-tolerance must be "
            "between 0 and 1"
        )

    if not 0.0 <= args.duplicate_rate_regression_tolerance <= 1.0:
        parser.error(
            "--duplicate-rate-regression-tolerance must "
            "be between 0 and 1"
        )

    if args.latency_regression_ratio < 0.0:
        parser.error("--latency-regression-ratio cannot be negative")
    if args.embedding_price_per_million_tokens < 0.0:
        parser.error("--embedding-price-per-million-tokens cannot be negative")
    if args.llm_input_price_per_million_tokens < 0.0:
        parser.error("--llm-input-price-per-million-tokens cannot be negative")
    args.cost_currency = args.cost_currency.strip()
    if not args.cost_currency:
        parser.error("--cost-currency cannot be empty")

    return args


def build_retrieval_evaluation_components(
) -> RetrievalEvaluationComponents:
    """组装 Baseline 与当前 v0.14 Candidate 共用的评估依赖。"""

    embedding_provider = EmbeddingFactory.create()
    chunk_embedding_repository = ChunkEmbeddingRepository()
    document_chunk_repository = DocumentChunkRepository()

    vector_store = DatabaseVectorStore(
        chunk_embedding_repository=chunk_embedding_repository
    )

    reranker = (
        RerankerFactory.create()
        if settings.reranker_enabled
        else None
    )

    retrieval_service = RetrievalService(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        default_top_k=settings.retrieval_top_k,
        default_candidate_k=settings.retrieval_candidate_k,
        default_score_threshold=settings.retrieval_score_threshold,
        default_per_document_limit=settings.retrieval_per_document_limit,
        document_chunk_repository=document_chunk_repository,
        # Baseline 在同一个 RetrievalService 中仍只检索 Parent，
        # Optimized 则检索 Child 并扩展回 Parent。
        parent_child_enabled=settings.parent_child_enabled,
        bm25_retriever=BM25RetrievalService(
            document_chunk_repository=document_chunk_repository,
        ),
        rrf_fusion_service=RRFFusionService(
            rank_constant=settings.retrieval_rrf_k,
        ),
        hybrid_enabled=settings.retrieval_hybrid_enabled,
        reranker=reranker,
        reranker_enabled=settings.reranker_enabled,
        reranker_fail_open=settings.reranker_fail_open,
    )

    return RetrievalEvaluationComponents(
        evaluator=RetrievalEvaluator(
            retrieval_service=retrieval_service,
        ),
        embedding_model=embedding_provider.model_name,
    )


def validate_v014_candidate_settings() -> None:
    """确保正式 v0.14 对比没有意外漏掉某一层候选策略。"""

    missing: list[str] = []

    if not settings.parent_child_enabled:
        missing.append("PARENT_CHILD_ENABLED=True")
    if not settings.retrieval_hybrid_enabled:
        missing.append("RETRIEVAL_HYBRID_ENABLED=True")
    if not settings.reranker_enabled:
        missing.append("RERANKER_ENABLED=True")

    if missing:
        raise RuntimeError(
            "v0.14 candidate profile is incomplete: "
            + ", ".join(missing)
        )


def build_dataset_validator(
) -> RetrievalDatasetValidator:
    """组装评估语料数据库校验器。"""

    return RetrievalDatasetValidator(
        document_repository=DocumentRepository(),
        document_content_repository=(
            DocumentContentRepository()
        ),
        document_chunk_repository=(
            DocumentChunkRepository()
        ),
    )


def build_configuration(
    args: argparse.Namespace,
    embedding_model: str,
) -> RetrievalEvaluationConfiguration:
    """生成不包含密钥的运行配置快照。"""

    return RetrievalEvaluationConfiguration(
        executed_at=datetime.now(timezone.utc),
        code_version=(
            args.code_version
            or resolve_code_version()
        ),
        vector_store_backend="database",
        embedding_provider=(
            settings.embedding_provider
        ),
        embedding_model=embedding_model,
        embedding_dimension=(
            settings.embedding_dimension
        ),
        shared_query_embedding_between_modes=True,
        chunk_strategy=settings.chunk_strategy,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        score_threshold=args.score_threshold,
        per_document_limit=(
            args.per_document_limit
        ),
        parent_child_enabled=settings.parent_child_enabled,
        child_chunk_size=(
            settings.parent_child_child_size
            if settings.parent_child_enabled
            else None
        ),
        child_chunk_overlap=(
            settings.parent_child_child_overlap
            if settings.parent_child_enabled
            else None
        ),
        hybrid_enabled=settings.retrieval_hybrid_enabled,
        rrf_k=(
            settings.retrieval_rrf_k
            if settings.retrieval_hybrid_enabled
            else None
        ),
        reranker_enabled=settings.reranker_enabled,
        reranker_model=(
            settings.reranker_model
            if settings.reranker_enabled
            else None
        ),
        reranker_fail_open=settings.reranker_fail_open,
        baseline_score_semantics="cosine_similarity",
        optimized_score_semantics=(
            "reranker_relevance"
            if settings.reranker_enabled
            else (
                "rrf_fusion"
                if settings.retrieval_hybrid_enabled
                else "cosine_similarity"
            )
        ),
    )


def build_token_cost_evaluator() -> RetrievalTokenCostEvaluator:
    """组装本地Token成本评估器。"""
    return RetrievalTokenCostEvaluator(
        document_content_repository=DocumentContentRepository(),
        document_chunk_repository=DocumentChunkRepository(),
    )


def build_token_cost_options(args: argparse.Namespace) -> TokenCostEvaluationOptions:
    """根据CLI参数构造Token成本估算配置。"""
    return TokenCostEvaluationOptions(
        currency=args.cost_currency,
        embedding_price_per_million_tokens=args.embedding_price_per_million_tokens,
        llm_input_price_per_million_tokens=args.llm_input_price_per_million_tokens,
    )


def build_regression_gate_thresholds(
    args: argparse.Namespace,
) -> RetrievalRegressionGateThresholds:
    """根据 CLI 参数构造回归门禁容忍度。"""

    return RetrievalRegressionGateThresholds(
        max_quality_metric_drop=(
            args.quality_regression_tolerance
        ),
        max_duplicate_rate_increase=(
            args.duplicate_rate_regression_tolerance
        ),
        max_latency_increase_ratio=(
            args.latency_regression_ratio
        ),
    )


def resolve_code_version() -> str | None:
    """尽力读取当前Git提交，不影响无Git环境运行。"""

    try:
        completed_process = subprocess.run(
            [
                "git",
                "rev-parse",
                "--short",
                "HEAD",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
    ):
        return None

    code_version = (
        completed_process.stdout.strip()
    )

    return code_version or None


def save_report(
    report: RetrievalComparisonReport,
    output_path: Path,
) -> None:
    """将评估报告保存为JSON文件。"""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_data = report.model_dump(
        mode="json"
    )

    output_path.write_text(
        json.dumps(
            report_data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def print_summary(
    report: RetrievalComparisonReport,
) -> None:
    """在终端输出两种模式的汇总指标。"""

    baseline = report.baseline.summary
    optimized = report.optimized.summary

    print()
    print(
        "Dataset: "
        f"{report.dataset.dataset_id} "
        f"v{report.dataset.dataset_version}"
    )
    print(
        "Dataset SHA256: "
        f"{report.dataset.source_sha256}"
    )
    print(
        "Code version: "
        f"{report.configuration.code_version or 'unknown'}"
    )
    print(
        "Shared query embedding: "
        f"{report.configuration.shared_query_embedding_between_modes}"
    )
    print()
    print("Retrieval evaluation completed.")
    print()

    _print_mode_summary(baseline)
    _print_mode_summary(optimized)
    _print_token_cost(report)
    _print_failure_analysis(report)
    _print_regression_gate(report)

    print("Metric changes:")
    _print_percentage_delta(
        "Overall Success Rate@K",
        optimized.hit_rate_at_k,
        baseline.hit_rate_at_k,
    )
    _print_percentage_delta(
        "Document Hit Rate@K",
        optimized.document_hit_rate_at_k,
        baseline.document_hit_rate_at_k,
    )
    _print_number_delta(
        "Document MRR",
        optimized.mean_reciprocal_rank,
        baseline.mean_reciprocal_rank,
    )
    _print_percentage_delta(
        "Document Recall@K",
        optimized.mean_document_coverage,
        baseline.mean_document_coverage,
    )
    _print_percentage_delta(
        "Chunk Hit Rate@K",
        optimized.chunk_hit_rate_at_k,
        baseline.chunk_hit_rate_at_k,
    )
    _print_number_delta(
        "Chunk MRR",
        optimized.mean_chunk_reciprocal_rank,
        baseline.mean_chunk_reciprocal_rank,
    )
    _print_percentage_delta(
        "Chunk Recall@K",
        optimized.mean_chunk_recall_at_k,
        baseline.mean_chunk_recall_at_k,
    )
    _print_number_delta(
        "Chunk nDCG@K",
        optimized.mean_chunk_ndcg_at_k,
        baseline.mean_chunk_ndcg_at_k,
    )
    _print_percentage_delta(
        "No-answer Accuracy",
        optimized.no_answer_accuracy,
        baseline.no_answer_accuracy,
    )
    _print_percentage_delta(
        "Duplicate Rate",
        optimized.mean_duplicate_rate,
        baseline.mean_duplicate_rate,
    )
    _print_number_delta(
        "Average Retrieval Latency",
        optimized.average_retrieval_latency_ms,
        baseline.average_retrieval_latency_ms,
        suffix=" ms",
    )
    _print_number_delta(
        "P95 Total Latency",
        optimized.p95_latency_ms,
        baseline.p95_latency_ms,
        suffix=" ms",
    )
    print()


def _print_mode_summary(
    summary: RetrievalEvaluationSummary,
) -> None:
    """输出单种检索模式的汇总指标。"""

    print(f"[{summary.retrieval_mode}]")
    print(f"  Cases: {summary.total_cases}")
    print(
        "  Answerable / No-answer / Chunk-labeled: "
        f"{summary.answerable_cases} / "
        f"{summary.no_answer_cases} / "
        f"{summary.chunk_labeled_cases}"
    )
    print(
        "  Overall Success Rate@K: "
        f"{summary.hit_rate_at_k:.2%}"
    )
    print(
        "  Document Hit Rate@K: "
        f"{summary.document_hit_rate_at_k:.2%}"
    )
    print(
        "  Document MRR: "
        f"{summary.mean_reciprocal_rank:.4f}"
    )
    print(
        "  Document Recall@K: "
        f"{summary.mean_document_coverage:.2%}"
    )
    print(
        "  Full Document Recall Rate@K: "
        f"{summary.full_document_coverage_rate_at_k:.2%}"
    )
    print(
        "  Chunk Hit Rate@K: "
        f"{summary.chunk_hit_rate_at_k:.2%}"
    )
    print(
        "  Chunk MRR: "
        f"{summary.mean_chunk_reciprocal_rank:.4f}"
    )
    print(
        "  Chunk Recall@K: "
        f"{summary.mean_chunk_recall_at_k:.2%}"
    )
    print(
        "  Chunk nDCG@K: "
        f"{summary.mean_chunk_ndcg_at_k:.4f}"
    )
    print(
        "  No-answer Accuracy: "
        f"{summary.no_answer_accuracy:.2%}"
    )
    print(
        "  No-answer False Positive Rate: "
        f"{summary.no_answer_false_positive_rate:.2%}"
    )
    print(
        "  Duplicate Rate: "
        f"{summary.mean_duplicate_rate:.2%}"
    )
    print(
        "  Expected Chunk Score Min / Mean: "
        f"{_format_optional_score(summary.minimum_first_expected_chunk_score)} / "
        f"{_format_optional_score(summary.mean_first_expected_chunk_score)}"
    )
    print(
        "  No-answer False-positive Score Max / Mean: "
        f"{_format_optional_score(summary.maximum_no_answer_false_positive_score)} / "
        f"{_format_optional_score(summary.mean_no_answer_false_positive_score)}"
    )
    print(
        "  Embedding / Retrieval / Total Avg Latency: "
        f"{summary.average_embedding_latency_ms:.2f} / "
        f"{summary.average_retrieval_latency_ms:.2f} / "
        f"{summary.average_latency_ms:.2f} ms"
    )
    print(
        "  P50 / P95 Total Latency: "
        f"{summary.p50_latency_ms:.2f} / "
        f"{summary.p95_latency_ms:.2f} ms"
    )
    print()


def _print_token_cost(report: RetrievalComparisonReport) -> None:
    """输出检索阶段Token与成本估算。"""
    usage = report.token_cost
    if usage is None:
        return

    ingestion = usage.ingestion
    print("Token / cost evaluation:")
    print(f"  Source / tokenizer: {usage.token_count_source} / {usage.tokenizer_name}")
    print(f"  Pricing: embedding={usage.pricing.embedding_price_per_million_tokens:.6f}, llm_input={usage.pricing.llm_input_price_per_million_tokens:.6f} {usage.pricing.currency} / 1M tokens")
    print(f"  Ingestion documents / chunks: {ingestion.document_count} / {ingestion.chunk_count}")
    print(f"  Source / embedded / overlap-extra tokens: {ingestion.source_tokens} / {ingestion.chunk_embedding_tokens} / {ingestion.estimated_overlap_extra_tokens} ({ingestion.estimated_overlap_overhead_rate:.2%})")
    print(f"  Chunk Avg / P50 / P95 tokens: {ingestion.average_chunk_tokens:.2f} / {ingestion.p50_chunk_tokens:.2f} / {ingestion.p95_chunk_tokens:.2f}")
    print(f"  Query embedding tokens total / avg / P95: {usage.total_query_embedding_tokens} / {usage.average_query_embedding_tokens:.2f} / {usage.p95_query_embedding_tokens:.2f}")
    print(f"  Estimated ingestion embedding cost: {ingestion.estimated_embedding_cost:.8f} {usage.pricing.currency}")
    _print_mode_token_cost("baseline", usage.baseline, usage.pricing.currency)
    _print_mode_token_cost("optimized", usage.optimized, usage.pricing.currency)
    print("  Note: costs are local estimates; context cost excludes system prompt, user question, chat history and LLM output.")
    print()


def _print_mode_token_cost(label, usage, currency: str) -> None:
    """输出单种检索模式的上下文Token与成本。"""
    print(f"  [{label}] Context total / avg / P50 / P95: {usage.total_context_tokens} / {usage.average_context_tokens:.2f} / {usage.p50_context_tokens:.2f} / {usage.p95_context_tokens:.2f}")
    print(f"  [{label}] Estimated retrieval-stage cost / query: {usage.estimated_average_cost_per_query:.8f} {currency}")
    print(f"  [{label}] Estimated cost per 1k / 10k queries: {usage.estimated_cost_per_1000_queries:.6f} / {usage.estimated_cost_per_10000_queries:.6f} {currency}")


def _print_failure_analysis(
    report: RetrievalComparisonReport,
) -> None:
    """输出关键失败与退化用例数量和代表问题。"""

    analysis = report.analysis

    print("Failure analysis:")
    print(
        "  Baseline chunk misses: "
        f"{analysis.baseline_chunk_miss_count}"
    )
    print(
        "  Optimized regressions: "
        f"{analysis.optimized_regression_count}"
    )
    print(
        "  Document gain + chunk loss: "
        f"{analysis.document_gain_chunk_loss_count}"
    )
    print(
        "  No-answer false positives: "
        f"{analysis.no_answer_false_positive_count}"
    )

    if analysis.optimized_regressions:
        print("  Top optimized regressions:")

        for comparison in analysis.optimized_regressions[:5]:
            reasons = ", ".join(comparison.reasons)
            print(
                "    - "
                f"{comparison.case_id}: {reasons}; "
                "chunk recall "
                f"{_format_optional_percentage(comparison.baseline_chunk_recall_at_k)} "
                "-> "
                f"{_format_optional_percentage(comparison.optimized_chunk_recall_at_k)}"
            )

    if analysis.no_answer_false_positives:
        print("  Highest no-answer false-positive scores:")

        for comparison in analysis.no_answer_false_positives[:5]:
            top_score = max(
                (
                    comparison.baseline_top_score
                    if comparison.baseline_top_score is not None
                    else -1.0
                ),
                (
                    comparison.optimized_top_score
                    if comparison.optimized_top_score is not None
                    else -1.0
                ),
            )
            print(
                "    - "
                f"{comparison.case_id}: top_score={top_score:.4f}"
            )

    print()


def _print_regression_gate(
    report: RetrievalComparisonReport,
) -> None:
    """输出候选策略回归门禁结果。"""

    gate = report.regression_gate
    status = "PASS" if gate.passed else "FAIL"

    print(f"Regression gate: {status}")
    print(
        "  Quality tolerance: "
        f"{gate.thresholds.max_quality_metric_drop:.2%}"
    )
    print(
        "  Duplicate-rate tolerance: "
        f"{gate.thresholds.max_duplicate_rate_increase:.2%}"
    )
    print(
        "  Latency increase tolerance: "
        f"{gate.thresholds.max_latency_increase_ratio:.2%}"
    )

    if gate.failed_metrics:
        print(
            "  Failed metrics: "
            + ", ".join(gate.failed_metrics)
        )

        failed_checks = [
            check
            for check in gate.checks
            if not check.passed
        ]

        for check in failed_checks:
            print(
                "    - "
                f"{check.metric}: "
                f"{check.baseline_value:.4f} -> "
                f"{check.optimized_value:.4f} "
                f"(delta={check.delta:+.4f})"
            )

    print()


def _format_optional_percentage(
    value: float | None,
) -> str:
    """格式化可能不存在的百分比指标。"""

    if value is None:
        return "n/a"

    return f"{value:.2%}"


def _format_optional_score(
    value: float | None,
) -> str:
    """格式化可能不存在的相似度分数。"""

    if value is None:
        return "n/a"

    return f"{value:.4f}"


def _print_percentage_delta(
    label: str,
    optimized_value: float,
    baseline_value: float,
) -> None:
    """输出百分比指标变化。"""

    print(
        f"  {label}: "
        f"{_format_delta(optimized_value, baseline_value, percentage=True)}"
    )


def _print_number_delta(
    label: str,
    optimized_value: float,
    baseline_value: float,
    suffix: str = "",
) -> None:
    """输出普通数值指标变化。"""

    print(
        f"  {label}: "
        f"{_format_delta(optimized_value, baseline_value, suffix=suffix)}"
    )


def _format_delta(
    optimized_value: float,
    baseline_value: float,
    percentage: bool = False,
    suffix: str = "",
) -> str:
    """格式化Optimized相对Baseline的变化。"""

    delta = optimized_value - baseline_value
    sign = "+" if delta >= 0 else ""

    if percentage:
        return f"{sign}{delta:.2%}"

    return f"{sign}{delta:.4f}{suffix}"


def main() -> None:
    """执行真实检索评估。"""

    args = parse_args()

    dataset = RetrievalCaseLoader.load(
        args.cases
    )
    dataset_reference = (
        RetrievalCaseLoader.build_reference(
            dataset=dataset,
            file_path=args.cases,
        )
    )

    dataset_validator = build_dataset_validator()

    with SessionLocal() as db:
        validation_result = (
            dataset_validator.validate(
                db=db,
                dataset=dataset,
            )
        )

        print(
            "Dataset validation passed: "
            f"documents={validation_result.corpus_document_count}, "
            f"chunks={validation_result.referenced_chunk_count}, "
            f"cases={validation_result.case_count}"
        )

        if args.validate_only:
            return

        if args.require_v014_candidate:
            validate_v014_candidate_settings()

        components = (
            build_retrieval_evaluation_components()
        )
        configuration = build_configuration(
            args=args,
            embedding_model=(
                components.embedding_model
            ),
        )

        report = components.evaluator.compare(
            db=db,
            cases=dataset.cases,
            dataset=dataset_reference,
            configuration=configuration,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            score_threshold=args.score_threshold,
            per_document_limit=args.per_document_limit,
            regression_gate_thresholds=build_regression_gate_thresholds(args),
        )
        token_cost = build_token_cost_evaluator().evaluate(
            db=db,
            dataset=dataset_reference,
            baseline=report.baseline,
            optimized=report.optimized,
            options=build_token_cost_options(args),
        )
        report = report.model_copy(update={"token_cost": token_cost})

    save_report(
        report=report,
        output_path=args.output,
    )

    print_summary(report)

    print(
        "Report saved to: "
        f"{args.output.resolve()}"
    )

    if (
        args.fail_on_regression
        and not report.regression_gate.passed
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
