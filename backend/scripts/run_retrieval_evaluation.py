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
from app.services.retrieval_service import (
    RetrievalService,
)
from app.services.vector_store.database import (
    DatabaseVectorStore,
)


DEFAULT_CASES_PATH = Path(
    "evaluation/retrieval_cases.json"
)

DEFAULT_REPORT_PATH = Path(
    "evaluation/reports/"
    "retrieval_comparison.json"
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
        help="Minimum cosine similarity score.",
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
        "--validate-only",
        action="store_true",
        help=(
            "Validate dataset structure and database "
            "references without calling Embedding or "
            "executing retrieval."
        ),
    )

    return parser.parse_args()


def build_retrieval_evaluation_components(
) -> RetrievalEvaluationComponents:
    """组装真实检索评估依赖。"""

    embedding_provider = EmbeddingFactory.create()

    chunk_embedding_repository = (
        ChunkEmbeddingRepository()
    )

    vector_store = DatabaseVectorStore(
        chunk_embedding_repository=(
            chunk_embedding_repository
        )
    )

    retrieval_service = RetrievalService(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        default_top_k=settings.retrieval_top_k,
        default_candidate_k=(
            settings.retrieval_candidate_k
        ),
        default_score_threshold=(
            settings.retrieval_score_threshold
        ),
        default_per_document_limit=(
            settings.retrieval_per_document_limit
        ),
    )

    return RetrievalEvaluationComponents(
        evaluator=RetrievalEvaluator(
            retrieval_service=retrieval_service,
        ),
        embedding_model=(
            embedding_provider.model_name
        ),
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
            per_document_limit=(
                args.per_document_limit
            ),
        )

    save_report(
        report=report,
        output_path=args.output,
    )

    print_summary(report)

    print(
        "Report saved to: "
        f"{args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
