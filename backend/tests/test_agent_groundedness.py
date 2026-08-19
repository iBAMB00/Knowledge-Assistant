from types import SimpleNamespace

import pytest
from sqlalchemy.orm import Session

from app.agent.context import ToolExecutionContext
from app.agent.evidence import build_knowledge_source_ref
from app.agent.model_response import LLMToolCall
from app.agent.native_agent import NativeAgentResult
from app.constants.user_role import UserRole
from app.models.database.document import Document
from app.models.database.document_chunk import DocumentChunk
from app.models.database.document_content import DocumentContent
from app.models.database.knowledge_base import KnowledgeBase
from app.models.database.user import User
from app.repositories.document_chunk_repository import DocumentChunkRepository
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
from app.services.evaluation.agent_evidence_loader import (
    AgentEvaluationEvidenceLoader,
    AgentEvidenceLoadResult,
)
from app.services.evaluation.agent_evaluator import AgentEvaluator
from app.services.evaluation.agent_live_evaluation_runner import (
    AgentLiveEvaluationRunner,
)
from app.services.evaluation.groundedness_judge import (
    GroundednessEvidence,
    GroundednessJudgeInvalidResponseError,
    GroundednessJudgeResult,
    OpenAICompatibleGroundednessJudge,
)


class _FakeCompletions:
    def __init__(self, content: str | None) -> None:
        self.content = content
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.content)
                )
            ]
        )


class _FakeClient:
    def __init__(self, content: str | None) -> None:
        self.completions = _FakeCompletions(content)
        self.chat = SimpleNamespace(completions=self.completions)


class _AllowAccessPolicy:
    def get_accessible_knowledge_base(self, **kwargs):
        return object()


class _FakeExecutionService:
    def __init__(
        self,
        *,
        answer: str,
        source_refs: list[str] | None = None,
    ) -> None:
        self.answer = answer
        self.source_refs = source_refs or []

    def run(self, *, db, context, message, observer):
        if self.source_refs:
            call = LLMToolCall(
                id="call-search",
                name="search_knowledge",
                arguments_json='{"query":"test"}',
            )
            observer.on_tool_call_requested(call)
            observer.on_tool_result(
                call_id=call.id,
                tool_name=call.name,
                ok=True,
                error_code=None,
                evidence_refs=self.source_refs,
            )
        observer.on_final_answer(self.answer)
        return NativeAgentResult(
            answer=self.answer,
            turns=1,
            tool_call_count=1 if self.source_refs else 0,
        )


class _StaticEvidenceLoader:
    def __init__(self, evidence: tuple[GroundednessEvidence, ...]) -> None:
        self.evidence = evidence

    def load(self, **kwargs) -> AgentEvidenceLoadResult:
        return AgentEvidenceLoadResult(
            evidence=self.evidence,
            missing_source_refs=(),
        )


class _RuleJudge:
    version = "test-judge-1.0"

    def judge(self, *, question, answer, evidence):
        if evidence:
            grounded = "不支持" not in answer
        else:
            grounded = "无法确认" in answer or "不知道" in answer
        return GroundednessJudgeResult(
            grounded=grounded,
            score=1.0 if grounded else 0.0,
            reason="deterministic test verdict",
        )


class _InvalidJudge:
    version = "test-judge-invalid"

    def judge(self, **kwargs):
        raise GroundednessJudgeInvalidResponseError("invalid")


def _dataset(*, query: str = "知识问题") -> AgentEvaluationDataset:
    return AgentEvaluationDataset(
        schema_version="1.0",
        dataset_id="groundedness-test",
        dataset_version="1.0.0",
        description="groundedness test",
        cases=[
            AgentEvaluationCase(
                case_id="search",
                query=query,
                category=AgentEvaluationCaseCategory.ONE_TOOL,
                expected_behavior="search and answer",
                allowed_tools=["search_knowledge"],
                forbidden_tools=[],
                expected_answerable=True,
                evaluate_groundedness=True,
                expected_tool_calls=[
                    AgentExpectedToolCall(tool_name="search_knowledge")
                ],
            )
        ],
    )


def _context() -> ToolExecutionContext:
    return ToolExecutionContext(
        user_id=1,
        role=UserRole.USER,
        knowledge_base_id=1,
        request_id="groundedness-test",
    )


def test_model_groundedness_judge_parses_supported_result() -> None:
    client = _FakeClient(
        '{"grounded":true,"score":0.97,"reason":"supported"}'
    )
    judge = OpenAICompatibleGroundednessJudge(
        client=client,
        model_name="judge-model",
    )

    result = judge.judge(
        question="Qdrant 如何部署？",
        answer="使用 Docker 部署。",
        evidence=[
            GroundednessEvidence(
                source_ref="doc:1:chunk:2",
                content="Qdrant 可以通过 Docker 运行。",
            )
        ],
    )

    assert result.grounded is True
    assert result.score == pytest.approx(0.97)
    system_prompt = client.completions.last_kwargs["messages"][0]["content"]
    assert "Citation identity is evaluated elsewhere" in system_prompt


def test_model_groundedness_judge_invalid_response_is_not_true() -> None:
    judge = OpenAICompatibleGroundednessJudge(
        client=_FakeClient("not-json"),
        model_name="judge-model",
    )

    with pytest.raises(GroundednessJudgeInvalidResponseError):
        judge.judge(
            question="问题",
            answer="回答",
            evidence=[],
        )


def test_evidence_loader_restores_parent_context_and_enforces_kb_scope(
    db: Session,
) -> None:
    user = User(
        email="grounded-loader@example.com",
        password_hash="hash",
        role=UserRole.USER.value,
        is_active=True,
    )
    db.add(user)
    db.flush()

    kb1 = KnowledgeBase(owner_id=user.id, name="KB1", description="one")
    kb2 = KnowledgeBase(owner_id=user.id, name="KB2", description="two")
    db.add_all([kb1, kb2])
    db.flush()

    def create_doc(kb_id: int, suffix: str):
        document = Document(
            knowledge_base_id=kb_id,
            filename=f"{suffix}.txt",
            storage_key=f"grounded/{suffix}.txt",
            size=10,
        )
        db.add(document)
        db.flush()
        content = DocumentContent(
            document_id=document.id,
            content="full",
            parser_type="text",
            parser_version="1.0",
        )
        db.add(content)
        db.flush()
        parent = DocumentChunk(
            document_content_id=content.id,
            chunk_index=0,
            content=f"parent-{suffix}",
        )
        db.add(parent)
        db.flush()
        child = DocumentChunk(
            document_content_id=content.id,
            chunk_index=1,
            content=f"child-{suffix}",
            parent_chunk_id=parent.id,
        )
        db.add(child)
        db.flush()
        return document, child

    doc1, child1 = create_doc(kb1.id, "one")
    doc2, child2 = create_doc(kb2.id, "two")

    loader = AgentEvaluationEvidenceLoader(
        chunk_repository=DocumentChunkRepository(),
        parent_child_enabled=True,
    )
    ref1 = build_knowledge_source_ref(
        document_id=doc1.id,
        chunk_id=child1.id,
    )
    ref2 = build_knowledge_source_ref(
        document_id=doc2.id,
        chunk_id=child2.id,
    )

    result = loader.load(
        db=db,
        knowledge_base_id=kb1.id,
        source_refs=[ref1, ref2],
    )

    assert [item.source_ref for item in result.evidence] == [ref1]
    assert result.evidence[0].content == "parent-one"
    assert result.missing_source_refs == (ref2,)


def test_live_runner_marks_supported_answer_grounded_without_persisting_text() -> None:
    source_ref = "doc:1:chunk:2"
    answer = f"部署方式有证据支持。[source:{source_ref}]"
    runner = AgentLiveEvaluationRunner(
        execution_service=_FakeExecutionService(
            answer=answer,
            source_refs=[source_ref],
        ),
        access_policy=_AllowAccessPolicy(),
        groundedness_judge=_RuleJudge(),
        evidence_loader=_StaticEvidenceLoader(
            (
                GroundednessEvidence(
                    source_ref=source_ref,
                    content="sensitive enterprise evidence",
                ),
            )
        ),
    )

    observations = runner.run_dataset(
        db=None,
        dataset=_dataset(),
        context=_context(),
    )
    observation = observations.observations[0]

    assert observations.runner_version == "1.2.0"
    assert observation.grounded is True
    assert observation.grounded_score == 1.0
    assert observation.grounded_judge_version == "test-judge-1.0"
    assert observation.grounded_judge_error_type is None
    serialized = observations.model_dump_json()
    assert answer not in serialized
    assert "sensitive enterprise evidence" not in serialized


def test_no_evidence_abstention_can_be_grounded() -> None:
    runner = AgentLiveEvaluationRunner(
        execution_service=_FakeExecutionService(
            answer="当前没有足够证据，无法确认。",
            source_refs=[],
        ),
        access_policy=_AllowAccessPolicy(),
        groundedness_judge=_RuleJudge(),
        evidence_loader=_StaticEvidenceLoader(()),
    )

    observation = runner.run_dataset(
        db=None,
        dataset=_dataset(query="不存在条款"),
        context=_context(),
    ).observations[0]

    assert observation.grounded is True
    assert observation.grounded_score == 1.0


def test_no_evidence_positive_claim_is_not_grounded() -> None:
    runner = AgentLiveEvaluationRunner(
        execution_service=_FakeExecutionService(
            answer="该条款明确规定必须在三天内完成。",
            source_refs=[],
        ),
        access_policy=_AllowAccessPolicy(),
        groundedness_judge=_RuleJudge(),
        evidence_loader=_StaticEvidenceLoader(()),
    )

    observation = runner.run_dataset(
        db=None,
        dataset=_dataset(query="不存在条款"),
        context=_context(),
    ).observations[0]

    assert observation.grounded is False
    assert observation.grounded_score == 0.0


def test_judge_failure_remains_unavailable_and_does_not_pass_task() -> None:
    source_ref = "doc:1:chunk:2"
    runner = AgentLiveEvaluationRunner(
        execution_service=_FakeExecutionService(
            answer=f"回答。[source:{source_ref}]",
            source_refs=[source_ref],
        ),
        access_policy=_AllowAccessPolicy(),
        groundedness_judge=_InvalidJudge(),
        evidence_loader=_StaticEvidenceLoader(
            (GroundednessEvidence(source_ref=source_ref, content="evidence"),)
        ),
    )
    dataset = _dataset()
    observations = runner.run_dataset(
        db=None,
        dataset=dataset,
        context=_context(),
    )
    observation = observations.observations[0]

    assert observation.grounded is None
    assert observation.grounded_judge_error_type == "invalid_response"

    report = AgentEvaluator().evaluate(
        dataset=dataset,
        dataset_reference=AgentEvaluationDatasetReference(
            schema_version="1.0",
            dataset_id=dataset.dataset_id,
            dataset_version=dataset.dataset_version,
            source_path="/tmp/test.json",
            source_sha256="a" * 64,
            total_cases=1,
        ),
        observations=observations,
    )
    assert report.cases[0].task_success is False
    assert report.summary.groundedness_coverage == 0.0
    assert report.summary.grounded_answer_rate is None


def test_valid_citation_does_not_override_semantically_unsupported_answer() -> None:
    source_ref = "doc:1:chunk:2"
    case = _dataset().cases[0]
    observation = AgentEvaluationObservation(
        case_id=case.case_id,
        run_succeeded=True,
        grounded=False,
        grounded_score=0.1,
        grounded_judge_version="judge-1",
        tool_calls=[
            AgentObservedToolCall(
                tool_name="search_knowledge",
                arguments={},
                error_code=None,
            )
        ],
        retrieved_sources=[source_ref],
        observed_sources=[source_ref],
        latency_ms=1.0,
    )

    result = AgentEvaluator().evaluate_case(
        case=case,
        observation=observation,
    )

    assert result.citation_correctness == 1.0
    assert result.grounded_answer is False
    assert result.task_success is False
