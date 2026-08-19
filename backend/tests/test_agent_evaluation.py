import json
from pathlib import Path

import pytest

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
from app.services.evaluation.agent_case_loader import AgentEvaluationCaseLoader
from app.services.evaluation.agent_evaluator import AgentEvaluator


ROOT = Path(__file__).resolve().parents[1]
AGENT_CASES_PATH = ROOT / "evaluation" / "agent_cases.json"


def _case(
    *,
    case_id: str = "case-1",
    category: AgentEvaluationCaseCategory = AgentEvaluationCaseCategory.ONE_TOOL,
    allowed_tools: list[str] | None = None,
    forbidden_tools: list[str] | None = None,
    expected_answerable: bool = True,
    expected_tool_calls: list[AgentExpectedToolCall] | None = None,
    expected_sources: list[str] | None = None,
    evaluate_groundedness: bool = False,
    require_retrieved_evidence: bool = False,
    require_citation: bool = False,
) -> AgentEvaluationCase:
    return AgentEvaluationCase(
        case_id=case_id,
        query="测试问题",
        category=category,
        expected_behavior="按标注执行",
        allowed_tools=allowed_tools or ["search_knowledge"],
        forbidden_tools=forbidden_tools or [],
        expected_sources=expected_sources or [],
        expected_answerable=expected_answerable,
        evaluate_groundedness=evaluate_groundedness,
        require_retrieved_evidence=require_retrieved_evidence,
        require_citation=require_citation,
        expected_tool_calls=(
            expected_tool_calls
            if expected_tool_calls is not None
            else [AgentExpectedToolCall(tool_name="search_knowledge")]
        ),
    )


def _reference(total_cases: int = 1) -> AgentEvaluationDatasetReference:
    return AgentEvaluationDatasetReference(
        schema_version="1.0",
        dataset_id="dataset",
        dataset_version="1.0.0",
        source_path="/tmp/agent_cases.json",
        source_sha256="a" * 64,
        total_cases=total_cases,
    )


def test_agent_cases_file_matches_v1_contract() -> None:
    dataset = AgentEvaluationCaseLoader.load_dataset(AGENT_CASES_PATH)

    assert dataset.schema_version == "1.0"
    assert dataset.dataset_id == "knowledge-assistant-agent-eval"
    assert len(dataset.cases) == 9
    assert dataset.dataset_version == "1.5.0"
    assert {
        case.category for case in dataset.cases
    } >= {
        AgentEvaluationCaseCategory.NO_TOOL,
        AgentEvaluationCaseCategory.ONE_TOOL,
        AgentEvaluationCaseCategory.MULTI_TOOL,
        AgentEvaluationCaseCategory.NO_ANSWER,
        AgentEvaluationCaseCategory.PERMISSION_DENIED,
        AgentEvaluationCaseCategory.TOOL_ERROR,
        AgentEvaluationCaseCategory.INJECTION,
    }

    by_id = {case.case_id: case for case in dataset.cases}
    assert (
        by_id["tool_error_processing_job"]
        .expected_tool_calls[0]
        .expected_error_code
        == "resource_not_found"
    )
    assert (
        by_id["permission_denied_cross_user_document"]
        .expected_tool_calls[0]
        .expected_error_code
        == "resource_not_found"
    )
    assert (
        by_id["one_tool_search_knowledge"]
        .expected_tool_calls[0]
        .expected_error_code
        is None
    )
    assert by_id["one_tool_search_knowledge"].require_retrieved_evidence is True
    assert by_id["one_tool_search_knowledge"].require_citation is True
    assert by_id["no_answer_search"].require_retrieved_evidence is False
    assert by_id["no_answer_search"].require_citation is False


def test_dataset_rejects_tool_outside_allowlist() -> None:
    with pytest.raises(ValueError, match="expected_tool_calls"):
        _case(
            allowed_tools=["search_knowledge"],
            expected_tool_calls=[
                AgentExpectedToolCall(tool_name="get_document")
            ],
        )


def test_no_tool_case_rejects_allowed_tool() -> None:
    with pytest.raises(ValueError, match="NO_TOOL"):
        _case(
            category=AgentEvaluationCaseCategory.NO_TOOL,
            allowed_tools=["search_knowledge"],
            expected_tool_calls=[],
        )


def test_loader_validates_dataset_and_observation_coverage(tmp_path: Path) -> None:
    dataset = AgentEvaluationDataset(
        schema_version="1.0",
        dataset_id="dataset",
        dataset_version="1.0.0",
        description="test",
        cases=[_case()],
    )
    case_path = tmp_path / "cases.json"
    case_path.write_text(dataset.model_dump_json(), encoding="utf-8")

    loaded = AgentEvaluationCaseLoader.load_dataset(case_path)
    reference = AgentEvaluationCaseLoader.build_reference(loaded, case_path)
    assert reference.source_sha256
    assert reference.total_cases == 1

    observation_set = AgentEvaluationObservationSet(
        dataset_id="dataset",
        dataset_version="1.0.0",
        observations=[
            AgentEvaluationObservation(
                case_id="case-1",
                run_succeeded=True,
                answerable=True,
                latency_ms=10,
            )
        ],
    )
    AgentEvaluationCaseLoader.validate_observation_coverage(
        loaded,
        observation_set,
    )

    wrong = observation_set.model_copy(
        update={"dataset_version": "2.0.0"}
    )
    with pytest.raises(ValueError, match="dataset_version"):
        AgentEvaluationCaseLoader.validate_observation_coverage(
            loaded,
            wrong,
        )


def test_evaluator_calculates_tool_selection_argument_and_cost_metrics() -> None:
    case = _case(
        expected_tool_calls=[
            AgentExpectedToolCall(
                tool_name="search_knowledge",
                expected_arguments={"top_k": 5},
            )
        ],
        expected_sources=["doc-a"],
    )
    dataset = AgentEvaluationDataset(
        schema_version="1.0",
        dataset_id="dataset",
        dataset_version="1.0.0",
        description="test",
        cases=[case],
    )
    observations = AgentEvaluationObservationSet(
        dataset_id="dataset",
        dataset_version="1.0.0",
        observations=[
            AgentEvaluationObservation(
                case_id="case-1",
                run_succeeded=True,
                answerable=True,
                grounded=True,
                tool_calls=[
                    AgentObservedToolCall(
                        tool_name="search_knowledge",
                        arguments={
                            "query": "Qdrant",
                            "top_k": 5,
                        },
                    )
                ],
                observed_sources=["doc-a"],
                latency_ms=125.5,
                input_tokens=100,
                output_tokens=40,
                cost=0.012,
            )
        ],
    )

    report = AgentEvaluator().evaluate(
        dataset=dataset,
        dataset_reference=_reference(),
        observations=observations,
    )

    result = report.cases[0]
    assert result.task_success is True
    assert result.tool_selection_pass is True
    assert result.tool_execution_pass is True
    assert result.tool_argument_accuracy == 1.0
    assert result.unnecessary_tool_call_rate == 0.0
    assert result.tool_policy_violation_count == 0
    assert result.citation_correctness == 1.0

    summary = report.summary
    assert summary.task_success_rate == 1.0
    assert summary.tool_selection_accuracy == 1.0
    assert summary.tool_execution_accuracy == 1.0
    assert summary.tool_argument_accuracy == 1.0
    assert summary.average_tool_calls == 1.0
    assert summary.total_input_tokens == 100
    assert summary.total_output_tokens == 40
    assert summary.total_cost == pytest.approx(0.012)


def test_evaluator_marks_policy_violation_and_unnecessary_tool_calls() -> None:
    case = _case(
        allowed_tools=["search_knowledge"],
        forbidden_tools=["get_document"],
    )
    observation = AgentEvaluationObservation(
        case_id="case-1",
        run_succeeded=True,
        answerable=True,
        tool_calls=[
            AgentObservedToolCall(
                tool_name="search_knowledge",
                arguments={},
            ),
            AgentObservedToolCall(
                tool_name="get_document",
                arguments={"document_id": 1},
            ),
        ],
        latency_ms=10,
    )

    result = AgentEvaluator().evaluate_case(
        case=case,
        observation=observation,
    )

    assert result.task_success is False
    assert result.tool_selection_pass is False
    assert result.tool_policy_violation_count == 1
    assert result.unnecessary_tool_call_rate == pytest.approx(0.5)


def test_argument_accuracy_is_none_without_argument_labels() -> None:
    case = _case()
    observation = AgentEvaluationObservation(
        case_id="case-1",
        run_succeeded=True,
        answerable=True,
        tool_calls=[
            AgentObservedToolCall(
                tool_name="search_knowledge",
                arguments={"query": "任意合理改写"},
            )
        ],
        latency_ms=10,
    )

    result = AgentEvaluator().evaluate_case(
        case=case,
        observation=observation,
    )

    assert result.tool_argument_accuracy is None
    assert result.task_success is True


def test_argument_accuracy_uses_subset_match() -> None:
    case = _case(
        expected_tool_calls=[
            AgentExpectedToolCall(
                tool_name="search_knowledge",
                expected_arguments={"document_id": 7},
            )
        ]
    )
    observation = AgentEvaluationObservation(
        case_id="case-1",
        run_succeeded=True,
        answerable=True,
        tool_calls=[
            AgentObservedToolCall(
                tool_name="search_knowledge",
                arguments={
                    "query": "部署",
                    "document_id": 7,
                    "top_k": 5,
                },
            )
        ],
        latency_ms=10,
    )

    result = AgentEvaluator().evaluate_case(
        case=case,
        observation=observation,
    )
    assert result.tool_argument_accuracy == 1.0



def test_expected_successful_tool_failure_fails_task_success() -> None:
    case = _case(
        expected_tool_calls=[
            AgentExpectedToolCall(tool_name="search_knowledge")
        ]
    )
    observation = AgentEvaluationObservation(
        case_id="case-1",
        run_succeeded=True,
        answerable=True,
        tool_calls=[
            AgentObservedToolCall(
                tool_name="search_knowledge",
                arguments={"query": "Qdrant"},
                error_code="execution_failed",
            )
        ],
        latency_ms=10,
    )

    result = AgentEvaluator().evaluate_case(
        case=case,
        observation=observation,
    )

    assert result.tool_selection_pass is True
    assert result.tool_execution_pass is False
    assert result.task_success is False


def test_expected_safe_tool_error_can_still_pass_task() -> None:
    case = _case(
        category=AgentEvaluationCaseCategory.TOOL_ERROR,
        allowed_tools=["get_processing_job"],
        expected_answerable=False,
        expected_tool_calls=[
            AgentExpectedToolCall(
                tool_name="get_processing_job",
                expected_arguments={"job_id": 999999},
                expected_error_code="resource_not_found",
            )
        ],
    )
    observation = AgentEvaluationObservation(
        case_id="case-1",
        run_succeeded=True,
        answerable=None,
        tool_calls=[
            AgentObservedToolCall(
                tool_name="get_processing_job",
                arguments={"job_id": 999999},
                error_code="resource_not_found",
            )
        ],
        latency_ms=10,
    )

    result = AgentEvaluator().evaluate_case(
        case=case,
        observation=observation,
    )

    assert result.tool_execution_pass is True
    assert result.tool_argument_accuracy == 1.0
    assert result.task_success is True


def test_extra_allowed_tool_call_fails_task_as_unnecessary() -> None:
    case = _case(
        allowed_tools=["search_knowledge"],
        expected_tool_calls=[
            AgentExpectedToolCall(tool_name="search_knowledge")
        ],
    )
    observation = AgentEvaluationObservation(
        case_id="case-1",
        run_succeeded=True,
        answerable=True,
        tool_calls=[
            AgentObservedToolCall(
                tool_name="search_knowledge",
                arguments={"query": "A"},
            ),
            AgentObservedToolCall(
                tool_name="search_knowledge",
                arguments={"query": "B"},
            ),
        ],
        latency_ms=10,
    )

    result = AgentEvaluator().evaluate_case(
        case=case,
        observation=observation,
    )

    assert result.tool_selection_pass is True
    assert result.tool_execution_pass is True
    assert result.unnecessary_tool_call_rate == pytest.approx(0.5)
    assert result.task_success is False

def test_evidence_requirement_rejects_successful_tool_with_zero_evidence() -> None:
    case = _case(
        evaluate_groundedness=True,
        require_retrieved_evidence=True,
        require_citation=True,
    )
    observation = AgentEvaluationObservation(
        case_id="case-1",
        run_succeeded=True,
        grounded=True,
        tool_calls=[
            AgentObservedToolCall(
                tool_name="search_knowledge",
                arguments={"query": "Qdrant"},
            )
        ],
        retrieved_sources=[],
        observed_sources=[],
        latency_ms=10,
    )

    result = AgentEvaluator().evaluate_case(case=case, observation=observation)

    assert result.tool_execution_pass is True
    assert result.grounded_answer is True
    assert result.retrieved_evidence_pass is False
    assert result.citation_requirement_pass is False
    assert result.citation_correctness is None
    assert result.task_success is False


def test_citation_requirement_rejects_missing_citation_after_retrieval() -> None:
    case = _case(
        evaluate_groundedness=True,
        require_retrieved_evidence=True,
        require_citation=True,
    )
    observation = AgentEvaluationObservation(
        case_id="case-1",
        run_succeeded=True,
        grounded=True,
        tool_calls=[AgentObservedToolCall(tool_name="search_knowledge")],
        retrieved_sources=["doc:1:chunk:2"],
        observed_sources=[],
        latency_ms=10,
    )

    result = AgentEvaluator().evaluate_case(case=case, observation=observation)

    assert result.retrieved_evidence_pass is True
    assert result.citation_requirement_pass is False
    assert result.citation_correctness == 0.0
    assert result.task_success is False


def test_positive_evidence_and_valid_citation_satisfy_contract() -> None:
    case = _case(
        evaluate_groundedness=True,
        require_retrieved_evidence=True,
        require_citation=True,
    )
    observation = AgentEvaluationObservation(
        case_id="case-1",
        run_succeeded=True,
        grounded=True,
        tool_calls=[AgentObservedToolCall(tool_name="search_knowledge")],
        retrieved_sources=["doc:1:chunk:2"],
        observed_sources=["doc:1:chunk:2"],
        latency_ms=10,
    )

    result = AgentEvaluator().evaluate_case(case=case, observation=observation)

    assert result.retrieved_evidence_pass is True
    assert result.citation_requirement_pass is True
    assert result.citation_correctness == 1.0
    assert result.task_success is True




def test_no_answer_case_is_not_penalized_for_not_citing_irrelevant_retrieval() -> None:
    case = _case(
        category=AgentEvaluationCaseCategory.NO_ANSWER,
        expected_answerable=False,
        evaluate_groundedness=True,
        require_retrieved_evidence=False,
        require_citation=False,
    )
    observation = AgentEvaluationObservation(
        case_id="case-1",
        run_succeeded=True,
        grounded=True,
        tool_calls=[AgentObservedToolCall(tool_name="search_knowledge")],
        retrieved_sources=["doc:1:chunk:2"],
        observed_sources=[],
        latency_ms=10,
    )

    result = AgentEvaluator().evaluate_case(case=case, observation=observation)

    assert result.retrieved_evidence_pass is None
    assert result.citation_requirement_pass is None
    assert result.citation_correctness is None
    assert result.task_success is True


def test_case_contract_rejects_citation_without_required_evidence() -> None:
    with pytest.raises(ValueError, match="require_citation"):
        _case(require_citation=True)


def test_case_contract_rejects_evidence_requirement_without_search_tool() -> None:
    with pytest.raises(ValueError, match="search_knowledge"):
        _case(
            allowed_tools=["get_document"],
            expected_tool_calls=[AgentExpectedToolCall(tool_name="get_document")],
            require_retrieved_evidence=True,
        )


def test_groundedness_is_not_invented_when_observation_missing() -> None:
    case = _case()
    observation = AgentEvaluationObservation(
        case_id="case-1",
        run_succeeded=True,
        answerable=True,
        tool_calls=[
            AgentObservedToolCall(
                tool_name="search_knowledge",
                arguments={},
            )
        ],
        latency_ms=10,
    )

    result = AgentEvaluator().evaluate_case(
        case=case,
        observation=observation,
    )

    assert result.grounded_answer is None
    assert result.citation_correctness is None


def test_loader_rejects_incomplete_observation_coverage() -> None:
    dataset = AgentEvaluationDataset(
        schema_version="1.0",
        dataset_id="dataset",
        dataset_version="1.0.0",
        description="test",
        cases=[_case(case_id="a"), _case(case_id="b")],
    )
    observations = AgentEvaluationObservationSet(
        dataset_id="dataset",
        dataset_version="1.0.0",
        observations=[
            AgentEvaluationObservation(
                case_id="a",
                run_succeeded=True,
                latency_ms=1,
            )
        ],
    )

    with pytest.raises(ValueError, match="missing"):
        AgentEvaluationCaseLoader.validate_observation_coverage(
            dataset,
            observations,
        )
