from app.agent.evidence import (
    build_knowledge_source_ref,
    extract_source_refs,
)
from app.schemas.agent_evaluation import (
    AgentEvaluationCase,
    AgentEvaluationCaseCategory,
    AgentEvaluationObservation,
)
from app.services.evaluation.agent_evaluator import AgentEvaluator


def _case(*, expected_sources: list[str] | None = None) -> AgentEvaluationCase:
    return AgentEvaluationCase(
        case_id="citation-case",
        query="根据知识库回答。",
        category=AgentEvaluationCaseCategory.ONE_TOOL,
        expected_behavior="检索后引用证据。",
        allowed_tools=["search_knowledge"],
        forbidden_tools=[],
        expected_sources=expected_sources or [],
        expected_answerable=True,
        expected_tool_calls=[{"tool_name": "search_knowledge"}],
    )


def _observation(
    *,
    retrieved_sources: list[str],
    observed_sources: list[str],
) -> AgentEvaluationObservation:
    return AgentEvaluationObservation(
        case_id="citation-case",
        run_succeeded=True,
        tool_calls=[
            {
                "tool_name": "search_knowledge",
                "arguments": {"query": "知识库"},
            }
        ],
        retrieved_sources=retrieved_sources,
        observed_sources=observed_sources,
        latency_ms=10,
    )


def test_source_ref_builder_and_answer_parser_are_stable() -> None:
    ref = build_knowledge_source_ref(document_id=12, chunk_id=34)

    assert ref == "doc:12:chunk:34"
    assert extract_source_refs(
        "A[source:doc:12:chunk:34] B[source:doc:12:chunk:34] "
        "C[source:doc:99:chunk:1]"
    ) == ["doc:12:chunk:34", "doc:99:chunk:1"]


def test_citation_correctness_uses_retrieved_evidence_without_ground_truth() -> None:
    result = AgentEvaluator().evaluate_case(
        case=_case(),
        observation=_observation(
            retrieved_sources=["doc:1:chunk:10", "doc:2:chunk:20"],
            observed_sources=["doc:1:chunk:10"],
        ),
    )

    assert result.citation_correctness == 1.0
    assert result.task_success is True


def test_missing_citation_fails_when_search_returned_evidence() -> None:
    result = AgentEvaluator().evaluate_case(
        case=_case(),
        observation=_observation(
            retrieved_sources=["doc:1:chunk:10"],
            observed_sources=[],
        ),
    )

    assert result.citation_correctness == 0.0
    assert result.task_success is False


def test_hallucinated_citation_is_not_counted_as_correct() -> None:
    result = AgentEvaluator().evaluate_case(
        case=_case(),
        observation=_observation(
            retrieved_sources=["doc:1:chunk:10"],
            observed_sources=["doc:999:chunk:999"],
        ),
    )

    assert result.citation_correctness == 0.0
    assert result.task_success is False


def test_dataset_expected_sources_take_priority_over_retrieved_sources() -> None:
    result = AgentEvaluator().evaluate_case(
        case=_case(expected_sources=["doc:2:chunk:20"]),
        observation=_observation(
            retrieved_sources=["doc:1:chunk:10", "doc:2:chunk:20"],
            observed_sources=["doc:1:chunk:10"],
        ),
    )

    assert result.citation_correctness == 0.0


def test_no_evidence_and_no_citation_remains_not_evaluable() -> None:
    result = AgentEvaluator().evaluate_case(
        case=_case(),
        observation=_observation(
            retrieved_sources=[],
            observed_sources=[],
        ),
    )

    assert result.citation_correctness is None
