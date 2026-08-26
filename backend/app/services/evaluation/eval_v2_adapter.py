from __future__ import annotations

from datetime import timezone

from app.schemas.agent_evaluation import AgentEvaluationReport
from app.schemas.evaluation_v2 import (
    EvaluationCaseResultV2,
    EvaluationDatasetIdentity,
    EvaluationDomain,
    EvaluationMetric,
    EvaluationMetricDirection,
    EvaluationMetricSource,
    EvaluationRunV2,
    EvaluationVersionIdentity,
)
from app.schemas.retrieval_evaluation import (
    RetrievalEvaluationDatasetReference,
    RetrievalEvaluationRun,
)
from app.schemas.stateful_fault_verification import StatefulFaultVerificationReport


class EvaluationV2Adapter:
    """
    将既有 Eval 证据转换成 Eval 2.0 统一只读视图。

    不重新计算原始指标，也不替代领域专用报告。
    """

    @classmethod
    def from_agent_report(
        cls,
        report: AgentEvaluationReport,
    ) -> EvaluationRunV2:
        summary = report.summary
        metrics = [
            cls._metric(
                "task_success_rate",
                summary.task_success_rate,
                higher=True,
            ),
            cls._metric(
                "tool_selection_accuracy",
                summary.tool_selection_accuracy,
                higher=True,
            ),
            cls._metric(
                "tool_execution_accuracy",
                summary.tool_execution_accuracy,
                higher=True,
                applicable=summary.tool_execution_accuracy is not None,
            ),
            cls._metric(
                "argument_correctness",
                summary.tool_argument_accuracy,
                higher=True,
                applicable=summary.tool_argument_accuracy is not None,
            ),
            cls._metric(
                "evidence_quality",
                summary.required_evidence_success_rate,
                higher=True,
                applicable=summary.required_evidence_success_rate is not None,
            ),
            cls._metric(
                "groundedness",
                summary.grounded_answer_rate,
                higher=True,
                source=EvaluationMetricSource.LLM_JUDGE,
                applicable=summary.grounded_answer_rate is not None,
            ),
            cls._metric(
                "citation",
                summary.required_citation_success_rate,
                higher=True,
                applicable=summary.required_citation_success_rate is not None,
            ),
            cls._metric(
                "citation_correctness",
                summary.citation_correctness,
                higher=True,
                applicable=summary.citation_correctness is not None,
            ),
            cls._metric(
                "safety_violation_count",
                summary.tool_policy_violation_count,
                higher=False,
                target_zero=True,
            ),
            cls._metric(
                "latency_ms",
                summary.average_latency_ms,
                higher=False,
                unit="ms",
            ),
            cls._metric(
                "cost",
                summary.total_cost,
                higher=False,
                applicable=summary.total_cost is not None,
            ),
        ]

        cases = tuple(
            EvaluationCaseResultV2(
                case_id=case.case_id,
                passed=case.task_success,
                trace_id=case.trace_id,
                provider_trace_id=case.provider_trace_id,
                agent_run_id=case.agent_run_id,
                metrics=(
                    cls._metric(
                        "task_success",
                        1 if case.task_success else 0,
                        higher=True,
                    ),
                    cls._metric(
                        "latency_ms",
                        case.latency_ms,
                        higher=False,
                        unit="ms",
                    ),
                ),
            )
            for case in report.cases
        )

        generated_at = report.generated_at
        if generated_at.tzinfo is None:
            generated_at = generated_at.replace(tzinfo=timezone.utc)

        return EvaluationRunV2(
            domain=EvaluationDomain.AGENT,
            generated_at=generated_at,
            dataset=EvaluationDatasetIdentity(
                dataset_id=report.dataset.dataset_id,
                dataset_version=report.dataset.dataset_version,
                source_path=report.dataset.source_path,
                source_sha256=report.dataset.source_sha256,
                total_cases=report.dataset.total_cases,
            ),
            version=EvaluationVersionIdentity(
                evaluator_version=report.evaluator_version,
                agent_version=report.agent_version,
                prompt_id=report.prompt_id,
                prompt_version=report.prompt_version,
            ),
            summary_metrics=tuple(metrics),
            cases=cases,
        )

    @classmethod
    def from_retrieval_run(
        cls,
        *,
        dataset: RetrievalEvaluationDatasetReference,
        run: RetrievalEvaluationRun,
        generated_at,
        retrieval_variant_id: str,
        code_version: str | None = None,
        average_context_tokens: float | None = None,
        reranker_tokens: int | None = None,
        reranker_provider_tokens: int | None = None,
        reranker_request_count: int | None = None,
    ) -> EvaluationRunV2:
        summary = run.summary
        metrics = [
            cls._metric("document_mrr", summary.mean_reciprocal_rank, higher=True),
            cls._metric("document_recall_at_k", summary.mean_document_coverage, higher=True),
            cls._metric("chunk_hit_rate_at_k", summary.chunk_hit_rate_at_k, higher=True),
            cls._metric("chunk_mrr", summary.mean_chunk_reciprocal_rank, higher=True),
            cls._metric("chunk_recall_at_k", summary.mean_chunk_recall_at_k, higher=True),
            cls._metric("chunk_ndcg_at_k", summary.mean_chunk_ndcg_at_k, higher=True),
            cls._metric("no_answer_accuracy", summary.no_answer_accuracy, higher=True),
            cls._metric(
                "retrieval_latency_ms",
                summary.average_retrieval_latency_ms,
                higher=False,
                unit="ms",
            ),
            cls._metric(
                "p95_total_latency_ms",
                summary.p95_latency_ms,
                higher=False,
                unit="ms",
            ),
        ]
        if average_context_tokens is not None:
            metrics.append(
                cls._metric(
                    "average_context_tokens",
                    average_context_tokens,
                    higher=False,
                    unit="tokens",
                )
            )
        resolved_reranker_tokens = (
            reranker_tokens
            if reranker_tokens is not None
            else reranker_provider_tokens
        )
        if resolved_reranker_tokens is not None:
            metrics.append(
                cls._metric(
                    "reranker_tokens",
                    resolved_reranker_tokens,
                    higher=False,
                    unit="tokens",
                )
            )
        if reranker_request_count is not None:
            metrics.append(
                cls._metric(
                    "reranker_request_count",
                    reranker_request_count,
                    higher=False,
                    unit="requests",
                )
            )

        cases = tuple(
            EvaluationCaseResultV2(
                case_id=case.case_id,
                passed=case.hit,
                metrics=(
                    cls._metric(
                        "chunk_recall_at_k",
                        case.chunk_recall_at_k,
                        higher=True,
                        applicable=case.chunk_recall_at_k is not None,
                    ),
                    cls._metric(
                        "latency_ms",
                        case.latency_ms,
                        higher=False,
                        unit="ms",
                    ),
                ),
            )
            for case in run.cases
        )

        return EvaluationRunV2(
            domain=EvaluationDomain.RETRIEVAL,
            generated_at=generated_at,
            dataset=EvaluationDatasetIdentity(
                dataset_id=dataset.dataset_id,
                dataset_version=dataset.dataset_version,
                source_path=dataset.source_path,
                source_sha256=dataset.source_sha256,
                total_cases=dataset.total_cases,
            ),
            version=EvaluationVersionIdentity(
                code_version=code_version,
                retrieval_variant_id=retrieval_variant_id,
            ),
            summary_metrics=tuple(metrics),
            cases=cases,
        )

    @classmethod
    def from_stateful_report(
        cls,
        report: StatefulFaultVerificationReport,
    ) -> EvaluationRunV2:
        summary = report.summary
        recovery_cases = [
            case
            for case in report.cases
            if case.category == "recovery"
        ]
        recovery_pass_rate = (
            sum(case.status == "pass" for case in recovery_cases)
            / len(recovery_cases)
            if recovery_cases
            else None
        )

        return EvaluationRunV2(
            domain=EvaluationDomain.STATEFUL,
            generated_at=report.generated_at,
            dataset=EvaluationDatasetIdentity(
                dataset_id=report.suite_id,
                dataset_version=report.suite_version,
                total_cases=summary.total_cases,
            ),
            version=EvaluationVersionIdentity(
                evaluator_version=report.runner_version,
                graph_version=report.graph_version,
            ),
            summary_metrics=(
                cls._metric("task_success_rate", summary.pass_rate, higher=True),
                cls._metric(
                    "recovery",
                    recovery_pass_rate,
                    higher=True,
                    applicable=recovery_pass_rate is not None,
                ),
                cls._metric(
                    "failed_cases",
                    summary.failed_cases,
                    higher=False,
                    target_zero=True,
                ),
            ),
            cases=tuple(
                EvaluationCaseResultV2(
                    case_id=case.case_id,
                    passed=case.status == "pass",
                    metrics=(
                        cls._metric(
                            "duration_ms",
                            case.duration_ms,
                            higher=False,
                            unit="ms",
                        ),
                    ),
                )
                for case in report.cases
            ),
        )

    @staticmethod
    def _metric(
        metric_id: str,
        value: float | int | None,
        *,
        higher: bool,
        source: EvaluationMetricSource = EvaluationMetricSource.DETERMINISTIC,
        unit: str | None = None,
        applicable: bool = True,
        target_zero: bool = False,
    ) -> EvaluationMetric:
        if target_zero:
            direction = EvaluationMetricDirection.TARGET_ZERO
        else:
            direction = (
                EvaluationMetricDirection.HIGHER_IS_BETTER
                if higher
                else EvaluationMetricDirection.LOWER_IS_BETTER
            )
        return EvaluationMetric(
            metric_id=metric_id,
            value=value,
            unit=unit,
            source=source,
            direction=direction,
            applicable=applicable,
        )
