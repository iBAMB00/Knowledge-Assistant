from collections.abc import Sequence
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from app.agent.context import ToolExecutionContext
from app.agent.model_response import LLMToolCall, LLMToolExchange, LLMToolResponse
from app.agent.native_agent import NativeAgentRunner
from app.agent.tools.base import BaseAgentTool, ToolContract, ToolRiskLevel
from app.constants.user_role import UserRole
from app.models.database.knowledge_base import KnowledgeBase
from app.models.database.user import User
from app.repositories.agent_run_repository import AgentRunRepository
from app.repositories.agent_tool_call_repository import AgentToolCallRepository
from app.repositories.document_repository import DocumentRepository
from app.repositories.knowledge_base_repository import KnowledgeBaseRepository
from app.schemas.agent_evaluation import (
    AgentEvaluationCase,
    AgentEvaluationCaseCategory,
    AgentEvaluationDataset,
    AgentEvaluationDatasetReference,
    AgentExpectedToolCall,
)
from app.services.agent_execution_service import AgentExecutionService
from app.services.evaluation.agent_evaluator import AgentEvaluator
from app.services.evaluation.agent_live_evaluation_runner import (
    AgentLiveEvaluationRunner,
)
from app.services.evaluation.agent_observation_collector import (
    AgentEvaluationObservationCollector,
)
from app.services.knowledge_base_access_policy import KnowledgeBaseAccessPolicy


class SearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    document_id: int | None = None


class SearchOutput(BaseModel):
    result_count: int
    source_ref: str


class SearchTool(BaseAgentTool[SearchInput, SearchOutput]):
    name = "search_knowledge"
    version = "1.0.0"
    description = "Eval search tool."
    risk_level = ToolRiskLevel.READ_ONLY
    input_model = SearchInput
    output_model = SearchOutput

    def execute(
        self,
        db: Session,
        context: ToolExecutionContext,
        tool_input: SearchInput,
    ) -> SearchOutput:
        return SearchOutput(
            result_count=1,
            source_ref="doc:7:chunk:70",
        )

    def extract_evidence_refs(self, output: SearchOutput) -> list[str]:
        return [output.source_ref]


class ScriptedLLM:
    def __init__(self, responses_by_message: dict[str, list[LLMToolResponse]]) -> None:
        self.responses_by_message = {
            key: list(value)
            for key, value in responses_by_message.items()
        }

    def chat_with_tool_history(
        self,
        *,
        message: str,
        tool_contracts: Sequence[ToolContract],
        history: Sequence[LLMToolExchange],
    ) -> LLMToolResponse:
        responses = self.responses_by_message[message]
        if not responses:
            raise AssertionError(f"unexpected extra model call: {message}")
        return responses.pop(0)


def _create_scope(db: Session) -> tuple[User, KnowledgeBase]:
    user = User(
        email="agent-eval-live@example.com",
        password_hash="hash",
        role=UserRole.USER.value,
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    kb = KnowledgeBase(
        owner_id=user.id,
        name="Agent Eval KB",
        description="D2 live eval",
    )
    db.add(kb)
    db.commit()
    db.refresh(kb)
    return user, kb


def _access_policy() -> KnowledgeBaseAccessPolicy:
    return KnowledgeBaseAccessPolicy(
        knowledge_base_repository=KnowledgeBaseRepository(),
        document_repository=DocumentRepository(),
    )


def _execution_service(llm: ScriptedLLM) -> AgentExecutionService:
    runner = NativeAgentRunner(
        llm_service=llm,
        tools=[SearchTool()],
    )
    return AgentExecutionService(
        agent_runner=runner,
        agent_run_repository=AgentRunRepository(),
        tool_call_repository=AgentToolCallRepository(),
        model_provider="test-provider",
        model_name="test-model",
    )


def _context(user: User, kb: KnowledgeBase) -> ToolExecutionContext:
    return ToolExecutionContext(
        user_id=user.id,
        role=UserRole.USER,
        knowledge_base_id=kb.id,
        request_id="agent-eval-bootstrap",
    )


def test_observation_collector_keeps_arguments_in_memory_and_safe_error_code() -> None:
    collector = AgentEvaluationObservationCollector()
    call = LLMToolCall(
        id="call-1",
        name="search_knowledge",
        arguments_json='{"query":"Qdrant","document_id":7}',
    )

    collector.on_tool_call_requested(call)
    collector.on_tool_result(
        call_id="call-1",
        tool_name="search_knowledge",
        ok=False,
        error_code="resource_not_found",
        evidence_refs=[],
    )

    observed = collector.build_tool_calls()
    assert observed[0].tool_name == "search_knowledge"
    assert observed[0].arguments == {
        "query": "Qdrant",
        "document_id": 7,
    }
    assert observed[0].error_code == "resource_not_found"


def test_live_eval_runner_executes_dataset_and_builds_deterministic_observations(
    db: Session,
) -> None:
    user, kb = _create_scope(db)
    dataset = AgentEvaluationDataset(
        schema_version="1.0",
        dataset_id="dataset",
        dataset_version="1.0.0",
        description="D2 test dataset",
        cases=[
            AgentEvaluationCase(
                case_id="direct",
                query="你能做什么？",
                category=AgentEvaluationCaseCategory.NO_TOOL,
                expected_behavior="直接回答",
                allowed_tools=[],
                forbidden_tools=["search_knowledge"],
                expected_answerable=True,
                expected_tool_calls=[],
            ),
            AgentEvaluationCase(
                case_id="search",
                query="查询文档 7",
                category=AgentEvaluationCaseCategory.ONE_TOOL,
                expected_behavior="调用 search_knowledge",
                allowed_tools=["search_knowledge"],
                forbidden_tools=[],
                expected_answerable=True,
                require_retrieved_evidence=True,
                require_citation=True,
                expected_tool_calls=[
                    AgentExpectedToolCall(
                        tool_name="search_knowledge",
                        expected_arguments={"document_id": 7},
                    )
                ],
            ),
        ],
    )
    llm = ScriptedLLM(
        {
            "你能做什么？": [LLMToolResponse(content="我可以帮助查询知识。")],
            "查询文档 7": [
                LLMToolResponse(
                    tool_calls=[
                        LLMToolCall(
                            id="call-search",
                            name="search_knowledge",
                            arguments_json=(
                                '{"query":"查询文档 7","document_id":7}'
                            ),
                        )
                    ]
                ),
                LLMToolResponse(
                    content="已根据工具结果回答。[source:doc:7:chunk:70]"
                ),
            ],
        }
    )

    observations = AgentLiveEvaluationRunner(
        execution_service=_execution_service(llm),
        access_policy=_access_policy(),
    ).run_dataset(
        db=db,
        dataset=dataset,
        context=_context(user, kb),
    )

    assert observations.runner_version == "1.2.0"
    assert observations.generated_at is not None
    assert len(observations.observations) == 2

    direct = observations.observations[0]
    assert direct.run_succeeded is True
    assert direct.run_error_type is None
    assert direct.tool_calls == []
    assert direct.answerable is None
    assert direct.grounded is None

    search = observations.observations[1]
    assert search.run_succeeded is True
    assert search.tool_calls[0].tool_name == "search_knowledge"
    assert search.tool_calls[0].arguments["document_id"] == 7
    assert search.tool_calls[0].error_code is None
    assert search.retrieved_sources == ["doc:7:chunk:70"]
    assert search.observed_sources == ["doc:7:chunk:70"]

    report = AgentEvaluator().evaluate(
        dataset=dataset,
        dataset_reference=AgentEvaluationDatasetReference(
            schema_version="1.0",
            dataset_id="dataset",
            dataset_version="1.0.0",
            source_path="/tmp/d2.json",
            source_sha256="a" * 64,
            total_cases=2,
        ),
        observations=observations,
    )
    assert report.summary.tool_selection_accuracy == 1.0
    assert report.summary.tool_execution_accuracy == 1.0
    assert report.summary.tool_argument_accuracy == 1.0
    assert report.summary.tool_policy_violation_count == 0
    assert report.summary.required_evidence_success_rate == 1.0
    assert report.summary.required_citation_success_rate == 1.0
    assert report.summary.citation_correctness == 1.0


def test_live_eval_records_requested_repeated_call_even_when_runtime_blocks_it(
    db: Session,
) -> None:
    user, kb = _create_scope(db)
    dataset = AgentEvaluationDataset(
        schema_version="1.0",
        dataset_id="dataset-repeat",
        dataset_version="1.0.0",
        description="repeat",
        cases=[
            AgentEvaluationCase(
                case_id="repeat",
                query="重复查询",
                category=AgentEvaluationCaseCategory.ONE_TOOL,
                expected_behavior="一次 search",
                allowed_tools=["search_knowledge"],
                forbidden_tools=[],
                expected_answerable=True,
                expected_tool_calls=[
                    AgentExpectedToolCall(tool_name="search_knowledge")
                ],
            )
        ],
    )
    same_args = '{"query":"重复查询"}'
    llm = ScriptedLLM(
        {
            "重复查询": [
                LLMToolResponse(
                    tool_calls=[
                        LLMToolCall(
                            id="call-1",
                            name="search_knowledge",
                            arguments_json=same_args,
                        )
                    ]
                ),
                LLMToolResponse(
                    tool_calls=[
                        LLMToolCall(
                            id="call-2",
                            name="search_knowledge",
                            arguments_json=same_args,
                        )
                    ]
                ),
            ]
        }
    )

    observations = AgentLiveEvaluationRunner(
        execution_service=_execution_service(llm),
        access_policy=_access_policy(),
    ).run_dataset(
        db=db,
        dataset=dataset,
        context=_context(user, kb),
    )

    observation = observations.observations[0]
    assert observation.run_succeeded is False
    assert observation.run_error_type == "repeated_tool_call"
    assert [call.tool_name for call in observation.tool_calls] == [
        "search_knowledge",
        "search_knowledge",
    ]


def test_live_eval_rejects_untrusted_or_inaccessible_kb_scope(
    db: Session,
) -> None:
    user, kb = _create_scope(db)
    other = User(
        email="agent-eval-other@example.com",
        password_hash="hash",
        role=UserRole.USER.value,
        is_active=True,
    )
    db.add(other)
    db.commit()
    db.refresh(other)

    other_kb = KnowledgeBase(
        owner_id=other.id,
        name="Other KB",
    )
    db.add(other_kb)
    db.commit()
    db.refresh(other_kb)

    dataset = AgentEvaluationDataset(
        schema_version="1.0",
        dataset_id="scope",
        dataset_version="1.0.0",
        description="scope",
        cases=[
            AgentEvaluationCase(
                case_id="direct",
                query="你好",
                category=AgentEvaluationCaseCategory.NO_TOOL,
                expected_behavior="direct",
                allowed_tools=[],
                forbidden_tools=["search_knowledge"],
                expected_answerable=True,
                expected_tool_calls=[],
            )
        ],
    )

    runner = AgentLiveEvaluationRunner(
        execution_service=_execution_service(
            ScriptedLLM({"你好": [LLMToolResponse(content="你好")]})
        ),
        access_policy=_access_policy(),
    )
    bad_context = ToolExecutionContext(
        user_id=user.id,
        role=UserRole.USER,
        knowledge_base_id=other_kb.id,
        request_id="eval-bad-scope",
    )

    with pytest.raises(ValueError, match="knowledge base not found"):
        runner.run_dataset(
            db=db,
            dataset=dataset,
            context=bad_context,
        )

    assert kb.id != other_kb.id
