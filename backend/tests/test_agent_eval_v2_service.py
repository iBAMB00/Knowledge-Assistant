from app.schemas.agent_evaluation import (
    AgentEvaluationCase,
    AgentEvaluationCaseCategory,
    AgentEvaluationDataset,
    AgentEvaluationDatasetReference,
    AgentEvaluationObservation,
    AgentEvaluationObservationSet,
    AgentExpectedToolCall,
    AgentObservedToolCall,
)
from app.services.evaluation.agent_eval_v2_service import AgentEvaluationV2Service


def _case() -> AgentEvaluationCase:
    return AgentEvaluationCase(
        case_id="case-1",
        query="q",
        category=AgentEvaluationCaseCategory.MULTI_TOOL,
        expected_behavior="first A then B",
        allowed_tools=["tool_a", "tool_b"],
        expected_answerable=True,
        expected_tool_calls=[
            AgentExpectedToolCall(tool_name="tool_a"),
            AgentExpectedToolCall(tool_name="tool_b"),
        ],
    )


def _dataset() -> AgentEvaluationDataset:
    return AgentEvaluationDataset(
        schema_version="1.0",
        dataset_id="dataset",
        dataset_version="1.0",
        description="test",
        cases=[_case()],
    )


def _reference() -> AgentEvaluationDatasetReference:
    return AgentEvaluationDatasetReference(
        schema_version="1.0",
        dataset_id="dataset",
        dataset_version="1.0",
        source_path="evaluation/agent.json",
        source_sha256="a" * 64,
        total_cases=1,
    )


def _observations(tool_names: list[str]) -> AgentEvaluationObservationSet:
    return AgentEvaluationObservationSet(
        dataset_id="dataset",
        dataset_version="1.0",
        observations=[
            AgentEvaluationObservation(
                case_id="case-1",
                run_succeeded=True,
                answerable=True,
                tool_calls=[
                    AgentObservedToolCall(tool_name=name)
                    for name in tool_names
                ],
                latency_ms=10,
            )
        ],
    )


def _metrics(run):
    return {metric.metric_id: metric.value for metric in run.summary_metrics}


def test_eval_v2_marks_wrong_tool_order_even_when_eval1_selection_passes() -> None:
    run = AgentEvaluationV2Service().evaluate(
        dataset=_dataset(),
        dataset_reference=_reference(),
        observations=_observations(["tool_b", "tool_a"]),
    )

    metrics = _metrics(run)
    assert metrics["tool_selection_accuracy"] == 1.0
    assert metrics["tool_sequence_accuracy"] == 0.0
    assert metrics["task_success_rate"] == 0.0
    assert run.cases[0].passed is False
    assert run.version.evaluator_version == "2.0.0"


def test_eval_v2_accepts_exact_tool_sequence() -> None:
    run = AgentEvaluationV2Service().evaluate(
        dataset=_dataset(),
        dataset_reference=_reference(),
        observations=_observations(["tool_a", "tool_b"]),
    )

    metrics = _metrics(run)
    assert metrics["tool_sequence_accuracy"] == 1.0
    assert metrics["task_success_rate"] == 1.0
    assert run.cases[0].passed is True
