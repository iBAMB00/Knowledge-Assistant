import argparse
import json
from pathlib import Path

from app.core.database import SessionLocal
from app.core.config import get_settings
from app.repositories.chunk_embedding_repository import (
    ChunkEmbeddingRepository,
)
from app.schemas.retrieval_evaluation import (
    RetrievalComparisonReport,
    RetrievalEvaluationSummary,
)
from app.services.embedding.factory import (
    EmbeddingFactory,
)
from app.services.evaluation.retrieval_case_loader import (
    RetrievalCaseLoader,
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



def parse_args() -> argparse.Namespace:
    """
    解析检索评估命令行参数。
    """

    parser = argparse.ArgumentParser(
        description=(
            "Compare baseline and optimized "
            "retrieval strategies."
        )
    )

    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_CASES_PATH,
        help=(
            "Path to the retrieval evaluation "
            "case JSON file."
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
        help=(
            "Minimum cosine similarity score."
        ),
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

    return parser.parse_args()


def build_retrieval_evaluator() -> RetrievalEvaluator:
    """
    组装真实检索评估依赖。
    """

    embedding_provider = (
        EmbeddingFactory.create()
    )

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
        default_candidate_k=settings.retrieval_candidate_k,
        default_score_threshold=settings.retrieval_score_threshold,
        default_per_document_limit=settings.retrieval_per_document_limit,
    )

    return RetrievalEvaluator(
        retrieval_service=retrieval_service,
    )


def save_report(
    report: RetrievalComparisonReport,
    output_path: Path,
) -> None:
    """
    将评估报告保存为JSON文件。
    """

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
    """
    在终端输出两种模式的汇总指标。
    """

    baseline = report.baseline.summary
    optimized = report.optimized.summary

    hit_rate_delta = _format_delta(
        optimized.hit_rate_at_k,
        baseline.hit_rate_at_k,
        percentage=True,
    )

    mrr_delta = _format_delta(
        optimized.mean_reciprocal_rank,
        baseline.mean_reciprocal_rank,
    )

    coverage_delta = _format_delta(
        optimized.mean_document_coverage,
        baseline.mean_document_coverage,
        percentage=True,
    )

    full_coverage_delta = _format_delta(
        optimized.full_document_coverage_rate_at_k,
        baseline.full_document_coverage_rate_at_k,
        percentage=True,
    )

    duplicate_delta = _format_delta(
        optimized.mean_duplicate_rate,
        baseline.mean_duplicate_rate,
        percentage=True,
    )

    latency_delta = _format_delta(
        optimized.average_latency_ms,
        baseline.average_latency_ms,
        suffix=" ms",
    )

    print()
    print("Retrieval evaluation completed.")
    print()

    _print_mode_summary(baseline)
    _print_mode_summary(optimized)

    print("Metric changes:")
    print(
        f"  Hit Rate@K: {hit_rate_delta}"
    )
    print(
        f"  MRR: {mrr_delta}"
    )
    print(
        "  Document Coverage: "
        f"{coverage_delta}"
    )
    print(
        "  Full Document Coverage Rate@K: "
        f"{full_coverage_delta}"
    )
    print(
        "  Duplicate Rate: "
        f"{duplicate_delta}"
    )
    print(
        "  Average Latency: "
        f"{latency_delta}"
    )
    print()


def _print_mode_summary(
    summary: RetrievalEvaluationSummary,
) -> None:
    """
    输出单种检索模式的汇总指标。
    """

    print(
        f"[{summary.retrieval_mode}]"
    )
    print(
        f"  Cases: {summary.total_cases}"
    )
    print(
        "  Hit Rate@K: "
        f"{summary.hit_rate_at_k:.2%}"
    )
    print(
        "  MRR: "
        f"{summary.mean_reciprocal_rank:.4f}"
    )
    print(
        "  Document Coverage: "
        f"{summary.mean_document_coverage:.2%}"
    )
    print(
        "  Full Document Coverage Rate@K: "
        f"{summary.full_document_coverage_rate_at_k:.2%}"
    )
    print(
        "  Duplicate Rate: "
        f"{summary.mean_duplicate_rate:.2%}"
    )
    print(
        "  Average Latency: "
        f"{summary.average_latency_ms:.2f} ms"
    )
    print()


def _format_delta(
    optimized_value: float,
    baseline_value: float,
    percentage: bool = False,
    suffix: str = "",
) -> str:
    """
    格式化Optimized相对Baseline的变化。
    """

    delta = (
        optimized_value - baseline_value
    )

    sign = "+" if delta >= 0 else ""

    if percentage:
        return f"{sign}{delta:.2%}"

    return (
        f"{sign}{delta:.4f}{suffix}"
    )


def main() -> None:
    """
    执行真实检索评估。
    """

    args = parse_args()

    cases = RetrievalCaseLoader.load(
        args.cases
    )

    evaluator = build_retrieval_evaluator()

    with SessionLocal() as db:
        report = evaluator.compare(
            db=db,
            cases=cases,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            score_threshold=(
                args.score_threshold
            ),
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