from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.schemas.agent_evaluation import (
    AgentEvaluationCaseCategory,
    AgentEvaluationCaseResult,
    AgentEvaluationDatasetReference,
    AgentEvaluationObservation,
    AgentEvaluationObservationSet,
    AgentEvaluationReport,
    AgentEvaluationSummary,
)
from app.services.evaluation.agent_runtime_comparison_service import (
    AgentRuntimeComparisonService,
)
from scripts import compare_agent_runtimes


DATASET_SHA = "a" * 64


def _dataset_reference(*, source_sha256: str = DATASET_SHA):
    return AgentEvaluationDatasetReference(
        schema_version="1.0",
        dataset_id="knowledge-assistant-agent-eval",
        dataset_version="1.5.0",
        source_path="evaluation/generated/agent_cases.bound.json",
        source_sha256=source_sha256,
        total_cases=2,
    )


def _summary(
    *,
    task_success_rate: float = 1.0,
    tool_selection_accuracy: float = 1.0,
    grounded_answer_rate: float | None = 1.0,
    groundedness_coverage: float | None = 1.0,
    average_tool_calls: float = 0.5,
    average_latency_ms: float = 100.0,
) -> AgentEvaluationSummary:
    return AgentEvaluationSummary(
        total_cases=2,
        task_success_rate=task_success_rate,
        tool_selection_accuracy=tool_selection_accuracy,
        tool_execution_accuracy=1.0,
        tool_argument_accuracy=1.0,
        unnecessary_tool_call_rate=0.0,
        tool_policy_violation_count=0,
        grounded_answer_rate=grounded_answer_rate,
        groundedness_coverage=groundedness_coverage,
        required_evidence_success_rate=1.0,
        required_citation_success_rate=1.0,
        citation_correctness=1.0,
        average_tool_calls=average_tool_calls,
        average_latency_ms=average_latency_ms,
        total_input_tokens=None,
        total_output_tokens=None,
        total_cost=None,
    )


def _case(
    *,
    case_id: str,
    category: AgentEvaluationCaseCategory,
    task_success: bool = True,
    tool_selection_pass: bool = True,
    tool_execution_pass: bool | None = True,
    groundedness_applicable: bool = False,
    grounded_answer: bool | None = None,
    groundedness_judge_error_type: str | None = None,
    tool_call_count: int = 1,
    latency_ms: float = 100.0,
) -> AgentEvaluationCaseResult:
    return AgentEvaluationCaseResult(
        case_id=case_id,
        category=category,
        task_success=task_success,
        tool_selection_pass=tool_selection_pass,
        tool_execution_pass=tool_execution_pass,
        tool_argument_accuracy=(
            1.0 if tool_execution_pass is not None else None
        ),
        unnecessary_tool_call_rate=0.0,
        tool_policy_violation_count=0,
        answerability_match=None,
        groundedness_applicable=groundedness_applicable,
        grounded_answer=grounded_answer,
        groundedness_score=(1.0 if grounded_answer is True else None),
        groundedness_judge_error_type=groundedness_judge_error_type,
        retrieved_evidence_pass=(
            True if groundedness_applicable else None
        ),
        citation_requirement_pass=(
            True if groundedness_applicable else None
        ),
        citation_correctness=(1.0 if groundedness_applicable else None),
        tool_call_count=tool_call_count,
        latency_ms=latency_ms,
        input_tokens=None,
        output_tokens=None,
        cost=None,
    )


def _report(
    *,
    summary: AgentEvaluationSummary | None = None,
    no_tool_task_success: bool = True,
    no_tool_selection_pass: bool = True,
    dataset_reference: AgentEvaluationDatasetReference | None = None,
) -> AgentEvaluationReport:
    return AgentEvaluationReport(
        generated_at=datetime.now(timezone.utc),
        evaluator_version="1.4.0",
        dataset=dataset_reference or _dataset_reference(),
        summary=summary or _summary(),
        cases=[
            _case(
                case_id="search_case",
                category=AgentEvaluationCaseCategory.ONE_TOOL,
                groundedness_applicable=True,
                grounded_answer=True,
            ),
            _case(
                case_id="no_tool_case",
                category=AgentEvaluationCaseCategory.NO_TOOL,
                task_success=no_tool_task_success,
                tool_selection_pass=no_tool_selection_pass,
                tool_execution_pass=None,
                tool_call_count=0,
                latency_ms=80.0,
            ),
        ],
    )


def _observations(
    *,
    runner_version: str,
    no_tool_run_succeeded: bool = True,
    no_tool_run_error_type: str | None = None,
) -> AgentEvaluationObservationSet:
    return AgentEvaluationObservationSet(
        dataset_id="knowledge-assistant-agent-eval",
        dataset_version="1.5.0",
        runner_version=runner_version,
        generated_at=datetime.now(timezone.utc),
        observations=[
            AgentEvaluationObservation(
                case_id="search_case",
                run_succeeded=True,
                run_error_type=None,
                answerable=None,
                grounded=True,
                grounded_score=1.0,
                grounded_judge_version="1.0.0",
                grounded_judge_error_type=None,
                tool_calls=[],
                retrieved_sources=["doc:1:chunk:2"],
                observed_sources=["doc:1:chunk:2"],
                latency_ms=100.0,
            ),
            AgentEvaluationObservation(
                case_id="no_tool_case",
                run_succeeded=no_tool_run_succeeded,
                run_error_type=no_tool_run_error_type,
                answerable=None,
                grounded=None,
                grounded_score=None,
                grounded_judge_version=None,
                grounded_judge_error_type=None,
                tool_calls=[],
                retrieved_sources=[],
                observed_sources=[],
                latency_ms=80.0,
            ),
        ],
    )


def _compare(
    *,
    baseline_report: AgentEvaluationReport | None = None,
    baseline_observations: AgentEvaluationObservationSet | None = None,
    candidate_report: AgentEvaluationReport | None = None,
    candidate_observations: AgentEvaluationObservationSet | None = None,
):
    return AgentRuntimeComparisonService().compare(
        baseline_report=baseline_report or _report(),
        baseline_observations=(
            baseline_observations
            or _observations(runner_version="native-v1:1.2.0")
        ),
        candidate_report=(
            candidate_report
            or _report(summary=_summary(average_latency_ms=75.0))
        ),
        candidate_observations=(
            candidate_observations
            or _observations(runner_version="langchain-v1:1.3.0")
        ),
    )


def test_runtime_comparison_passes_without_quality_regression() -> None:
    comparison = _compare()

    assert comparison.summary.decision == "pass"
    assert comparison.summary.deterministic_gate_passed is True
    assert comparison.summary.groundedness_gate_status == "pass"
    assert comparison.summary.failed_metrics == []
    assert comparison.summary.regression_case_ids == []
    assert comparison.summary.average_latency_ms_delta == pytest.approx(-25.0)
    assert comparison.summary.average_latency_ratio == pytest.approx(0.75)
    assert comparison.baseline_runner_version == "native-v1:1.2.0"
    assert comparison.candidate_runner_version == "langchain-v1:1.3.0"


def test_runtime_comparison_fails_when_candidate_aggregate_quality_regresses() -> None:
    comparison = _compare(
        candidate_report=_report(
            summary=_summary(tool_selection_accuracy=0.5)
        )
    )

    assert comparison.summary.decision == "fail"
    assert comparison.summary.deterministic_gate_passed is False
    assert "tool_selection_accuracy" in comparison.summary.failed_metrics


def test_runtime_comparison_detects_per_case_run_regression_even_when_summary_matches() -> None:
    comparison = _compare(
        candidate_report=_report(no_tool_task_success=False),
        candidate_observations=_observations(
            runner_version="langchain-v1:1.3.0",
            no_tool_run_succeeded=False,
            no_tool_run_error_type="agent_timeout",
        ),
    )

    assert comparison.summary.decision == "fail"
    assert comparison.summary.regression_case_ids == ["no_tool_case"]
    case = next(
        item
        for item in comparison.case_comparisons
        if item.case_id == "no_tool_case"
    )
    assert case.regression_reasons == ["run_succeeded_lost"]


def test_runtime_comparison_is_inconclusive_when_groundedness_coverage_is_incomplete() -> None:
    candidate_summary = _summary(
        task_success_rate=0.5,
        grounded_answer_rate=1.0,
        groundedness_coverage=0.5,
    )
    candidate_report = _report(summary=candidate_summary)
    comparison = _compare(candidate_report=candidate_report)

    assert comparison.summary.decision == "inconclusive"
    assert comparison.summary.deterministic_gate_passed is True
    assert comparison.summary.groundedness_gate_status == "inconclusive"
    assert comparison.summary.inconclusive_metrics == [
        "grounded_answer_rate"
    ]
    # Task Success 只展示差值，不因 Judge coverage 不完整误判 Runtime 回归。
    task_check = next(
        check
        for check in comparison.metric_checks
        if check.metric == "task_success_rate"
    )
    assert task_check.status == "informational"
    assert comparison.summary.task_success_rate_delta == pytest.approx(-0.5)



def test_runtime_comparison_is_inconclusive_when_required_candidate_metric_is_missing() -> None:
    candidate_summary = _summary().model_copy(
        update={"tool_execution_accuracy": None}
    )
    comparison = _compare(
        candidate_report=_report(summary=candidate_summary)
    )

    assert comparison.summary.decision == "inconclusive"
    assert comparison.summary.deterministic_gate_passed is False
    assert "tool_execution_accuracy" in comparison.summary.inconclusive_metrics

def test_runtime_comparison_rejects_different_dataset_snapshot() -> None:
    candidate_report = _report(
        dataset_reference=_dataset_reference(source_sha256="b" * 64)
    )

    with pytest.raises(ValueError, match="same dataset snapshot"):
        _compare(candidate_report=candidate_report)


def test_compare_agent_runtimes_script_writes_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    baseline_report_path = tmp_path / "native_report.json"
    baseline_observation_path = tmp_path / "native_observations.json"
    candidate_report_path = tmp_path / "langchain_report.json"
    candidate_observation_path = tmp_path / "langchain_observations.json"
    output_path = tmp_path / "comparison.json"

    baseline_report_path.write_text(
        _report().model_dump_json(indent=2),
        encoding="utf-8",
    )
    baseline_observation_path.write_text(
        _observations(
            runner_version="native-v1:1.2.0"
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )
    candidate_report_path.write_text(
        _report(
            summary=_summary(average_latency_ms=75.0)
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )
    candidate_observation_path.write_text(
        _observations(
            runner_version="langchain-v1:1.3.0"
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_agent_runtimes",
            "--baseline-report",
            str(baseline_report_path),
            "--baseline-observations",
            str(baseline_observation_path),
            "--candidate-report",
            str(candidate_report_path),
            "--candidate-observations",
            str(candidate_observation_path),
            "--output",
            str(output_path),
        ],
    )

    assert compare_agent_runtimes.main() == 0
    rendered = json.loads(capsys.readouterr().out)
    assert rendered["decision"] == "pass"
    assert rendered["baseline_runner_version"] == "native-v1:1.2.0"
    assert rendered["candidate_runner_version"] == "langchain-v1:1.3.0"

    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    assert persisted["summary"]["decision"] == "pass"
