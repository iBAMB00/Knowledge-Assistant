from datetime import datetime, timezone

from app.schemas.agent_evaluation import (
    AgentEvaluationCaseCategory,
    AgentEvaluationCaseResult,
    AgentEvaluationDatasetReference,
    AgentEvaluationReport,
    AgentEvaluationSummary,
)
from app.schemas.evaluation_v2 import (
    EvaluationDomain,
    EvaluationMetricSource,
)
from app.schemas.retrieval_evaluation import (
    RetrievalCaseCategory,
    RetrievalCaseDifficulty,
    RetrievalEvaluationCaseResult,
    RetrievalEvaluationDatasetReference,
    RetrievalEvaluationRun,
    RetrievalEvaluationSummary,
)
from app.schemas.stateful_fault_verification import (
    StatefulFaultCaseResult,
    StatefulFaultVerificationReport,
    StatefulFaultVerificationSummary,
)
from app.services.evaluation.eval_v2_adapter import EvaluationV2Adapter


def _metric_map(run):
    return {metric.metric_id: metric for metric in run.summary_metrics}


def test_agent_report_adapts_prompt_trace_and_quality_metrics() -> None:
    now = datetime.now(timezone.utc)
    report = AgentEvaluationReport(
        generated_at=now,
        evaluator_version="2.0-test",
        agent_version="agent-v1",
        prompt_id="agent.tool-calling-system",
        prompt_version="1.1.0",
        dataset=AgentEvaluationDatasetReference(
            schema_version="1.0",
            dataset_id="agent-dataset",
            dataset_version="1.0",
            source_path="evaluation/agent.json",
            source_sha256="a" * 64,
            total_cases=1,
        ),
        summary=AgentEvaluationSummary(
            total_cases=1,
            task_success_rate=1.0,
            tool_selection_accuracy=1.0,
            tool_execution_accuracy=1.0,
            tool_argument_accuracy=1.0,
            unnecessary_tool_call_rate=0.0,
            tool_policy_violation_count=0,
            grounded_answer_rate=1.0,
            groundedness_coverage=1.0,
            required_evidence_success_rate=1.0,
            required_citation_success_rate=1.0,
            citation_correctness=1.0,
            average_tool_calls=1.0,
            average_latency_ms=120.0,
            total_input_tokens=100,
            total_output_tokens=20,
            total_cost=0.01,
        ),
        cases=[
            AgentEvaluationCaseResult(
                case_id="case-1",
                trace_id="trace-1",
                provider_trace_id="provider-1",
                agent_run_id=7,
                category=AgentEvaluationCaseCategory.ONE_TOOL,
                task_success=True,
                tool_selection_pass=True,
                tool_execution_pass=True,
                tool_argument_accuracy=1.0,
                unnecessary_tool_call_rate=0.0,
                tool_policy_violation_count=0,
                groundedness_applicable=True,
                grounded_answer=True,
                groundedness_score=1.0,
                retrieved_evidence_pass=True,
                citation_requirement_pass=True,
                citation_correctness=1.0,
                tool_call_count=1,
                latency_ms=120.0,
                input_tokens=100,
                output_tokens=20,
                cost=0.01,
            )
        ],
    )

    run = EvaluationV2Adapter.from_agent_report(report)
    metrics = _metric_map(run)

    assert run.domain == EvaluationDomain.AGENT
    assert run.version.prompt_id == "agent.tool-calling-system"
    assert run.version.prompt_version == "1.1.0"
    assert run.cases[0].trace_id == "trace-1"
    assert metrics["task_success_rate"].value == 1.0
    assert metrics["groundedness"].source == EvaluationMetricSource.LLM_JUDGE
    assert metrics["safety_violation_count"].value == 0


def test_retrieval_run_adapts_quality_and_latency_metrics() -> None:
    summary = RetrievalEvaluationSummary(
        retrieval_mode="optimized",
        total_cases=1,
        answerable_cases=1,
        no_answer_cases=0,
        chunk_labeled_cases=1,
        hit_rate_at_k=1.0,
        document_hit_rate_at_k=1.0,
        mean_reciprocal_rank=1.0,
        mean_document_coverage=1.0,
        full_document_coverage_rate_at_k=1.0,
        chunk_hit_rate_at_k=1.0,
        mean_chunk_reciprocal_rank=0.5,
        mean_chunk_recall_at_k=1.0,
        mean_chunk_ndcg_at_k=0.8,
        mean_duplicate_rate=0.0,
        no_answer_accuracy=0.0,
        no_answer_false_positive_rate=0.0,
        minimum_first_expected_chunk_score=0.8,
        mean_first_expected_chunk_score=0.8,
        maximum_no_answer_false_positive_score=None,
        mean_no_answer_false_positive_score=None,
        average_embedding_latency_ms=10.0,
        average_retrieval_latency_ms=50.0,
        average_latency_ms=60.0,
        p50_latency_ms=60.0,
        p95_latency_ms=70.0,
    )
    case = RetrievalEvaluationCaseResult(
        case_id="case-1",
        question="q",
        category=RetrievalCaseCategory.EXACT_TERM,
        difficulty=RetrievalCaseDifficulty.EASY,
        should_retrieve=True,
        retrieval_mode="optimized",
        expected_document_ids=[1],
        expected_chunk_ids=[10],
        retrieved_document_ids=[1],
        retrieved_chunk_ids=[10],
        retrieved_results=[],
        hit=True,
        reciprocal_rank=1.0,
        document_coverage=1.0,
        document_hit_at_k=True,
        chunk_hit_at_k=True,
        chunk_reciprocal_rank=0.5,
        chunk_recall_at_k=1.0,
        chunk_ndcg_at_k=0.8,
        top_score=0.9,
        first_expected_document_score=0.9,
        first_expected_chunk_score=0.9,
        duplicate_rate=0.0,
        no_answer_false_positive=False,
        embedding_latency_ms=10.0,
        retrieval_latency_ms=50.0,
        latency_ms=60.0,
    )
    run = EvaluationV2Adapter.from_retrieval_run(
        dataset=RetrievalEvaluationDatasetReference(
            schema_version="1.0",
            dataset_id="retrieval-dataset",
            dataset_version="2.0",
            source_path="evaluation/retrieval.json",
            source_sha256="b" * 64,
            strict_corpus=True,
            corpus_document_ids=[1],
            total_cases=1,
        ),
        run=RetrievalEvaluationRun(summary=summary, cases=[case]),
        generated_at=datetime.now(timezone.utc),
        retrieval_variant_id="full",
        code_version="abc",
        average_context_tokens=123.0,
        reranker_provider_tokens=456,
        reranker_request_count=7,
    )
    metrics = _metric_map(run)

    assert run.domain == EvaluationDomain.RETRIEVAL
    assert run.version.retrieval_variant_id == "full"
    assert metrics["chunk_mrr"].value == 0.5
    assert metrics["chunk_ndcg_at_k"].value == 0.8
    assert metrics["retrieval_latency_ms"].value == 50.0
    assert metrics["average_context_tokens"].value == 123.0
    assert metrics["reranker_provider_tokens"].value == 456
    assert metrics["reranker_request_count"].value == 7


def test_stateful_report_adapts_recovery_metric() -> None:
    now = datetime.now(timezone.utc)
    report = StatefulFaultVerificationReport(
        generated_at=now,
        suite_id="stateful-suite",
        suite_version="1.0",
        runner_version="runner-1",
        graph_version="graph-1",
        checkpoint_schema_version="1",
        state_schema_version="1",
        pytest_exit_code=0,
        summary=StatefulFaultVerificationSummary(
            decision="pass",
            total_cases=2,
            passed_cases=2,
            failed_cases=0,
            skipped_cases=0,
            pass_rate=1.0,
        ),
        cases=(
            StatefulFaultCaseResult(
                case_id="recover-1",
                category="recovery",
                description="recover",
                status="pass",
                duration_ms=10,
                nodeids=("tests/test_x.py::test_x",),
            ),
            StatefulFaultCaseResult(
                case_id="graph-1",
                category="graph",
                description="graph",
                status="pass",
                duration_ms=5,
                nodeids=("tests/test_y.py::test_y",),
            ),
        ),
    )

    run = EvaluationV2Adapter.from_stateful_report(report)
    metrics = _metric_map(run)

    assert run.domain == EvaluationDomain.STATEFUL
    assert metrics["recovery"].value == 1.0
    assert metrics["failed_cases"].value == 0
