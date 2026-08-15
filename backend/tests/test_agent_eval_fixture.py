from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from app.models.database.processing_job import ProcessingJob
from app.repositories.document_repository import DocumentRepository
from app.repositories.knowledge_base_repository import KnowledgeBaseRepository
from app.repositories.user_repository import UserRepository
from app.schemas.agent_evaluation import (
    AgentEvaluationCase,
    AgentEvaluationCaseCategory,
    AgentEvaluationDataset,
    AgentEvaluationFixtureManifest,
    AgentExpectedToolCall,
)
from app.services.evaluation.agent_case_loader import AgentEvaluationCaseLoader
from app.services.evaluation.agent_dataset_binder import AgentEvaluationDatasetBinder
from app.services.evaluation.agent_eval_fixture_service import (
    AgentEvaluationFixtureService,
)
from scripts.prepare_agent_live_evaluation import _ensure_safe_environment


ROOT = Path(__file__).resolve().parents[1]
AGENT_CASES_PATH = ROOT / "evaluation" / "agent_cases.json"


def _fixture_service() -> AgentEvaluationFixtureService:
    return AgentEvaluationFixtureService(
        user_repository=UserRepository(),
        knowledge_base_repository=KnowledgeBaseRepository(),
        document_repository=DocumentRepository(),
    )


def _manifest(**updates) -> AgentEvaluationFixtureManifest:
    values = {
        "schema_version": "1.0",
        "fixture_version": "1.0.0",
        "generated_at": "2026-08-16T00:00:00Z",
        "primary_user_id": 10,
        "primary_role": "user",
        "primary_knowledge_base_id": 20,
        "primary_document_id": 101,
        "cross_user_id": 11,
        "cross_user_knowledge_base_id": 21,
        "cross_user_document_id": 202,
        "missing_processing_job_id": 303,
        "bindings": {
            "primary_document_id": 101,
            "cross_user_document_id": 202,
            "missing_processing_job_id": 303,
        },
    }
    values.update(updates)
    return AgentEvaluationFixtureManifest.model_validate(values)


def test_prepare_fixture_is_idempotent_and_preserves_cross_user_boundary(
    db: Session,
) -> None:
    service = _fixture_service()

    first = service.prepare(db=db)
    second = service.prepare(db=db)

    assert second.primary_user_id == first.primary_user_id
    assert second.primary_knowledge_base_id == first.primary_knowledge_base_id
    assert second.primary_document_id == first.primary_document_id
    assert second.cross_user_document_id == first.cross_user_document_id

    assert first.primary_user_id != first.cross_user_id
    assert first.primary_knowledge_base_id != first.cross_user_knowledge_base_id

    primary_document = DocumentRepository().find_by_id(
        db=db,
        document_id=first.primary_document_id,
    )
    cross_document = DocumentRepository().find_by_id(
        db=db,
        document_id=first.cross_user_document_id,
    )

    assert primary_document is not None
    assert cross_document is not None
    assert primary_document.knowledge_base_id == first.primary_knowledge_base_id
    assert (
        cross_document.knowledge_base_id
        == first.cross_user_knowledge_base_id
    )
    assert db.get(ProcessingJob, first.missing_processing_job_id) is None


def test_fixture_users_are_reserved_non_login_identities(db: Session) -> None:
    manifest = _fixture_service().prepare(db=db)
    users = UserRepository()

    primary = users.find_by_id(db=db, user_id=manifest.primary_user_id)
    cross_user = users.find_by_id(db=db, user_id=manifest.cross_user_id)

    assert primary is not None
    assert cross_user is not None
    assert primary.email.endswith("@fixture.invalid")
    assert cross_user.email.endswith("@fixture.invalid")
    assert primary.password_hash == "agent-eval-fixture-no-login"
    assert cross_user.password_hash == "agent-eval-fixture-no-login"


def test_dataset_binder_replaces_ids_in_query_notes_and_arguments() -> None:
    dataset = AgentEvaluationDataset(
        schema_version="1.0",
        dataset_id="dataset",
        dataset_version="1.0.0",
        description="fixture binding",
        fixture_placeholders={
            "primary_document_id": 1001,
            "cross_user_document_id": 999998,
            "missing_processing_job_id": 999999,
        },
        cases=[
            AgentEvaluationCase(
                case_id="document",
                query="查看文档 1001；不要访问 999998。",
                category=AgentEvaluationCaseCategory.ONE_TOOL,
                expected_behavior="读取当前文档",
                allowed_tools=["get_document"],
                forbidden_tools=[],
                expected_answerable=True,
                expected_tool_calls=[
                    AgentExpectedToolCall(
                        tool_name="get_document",
                        expected_arguments={"document_id": 1001},
                    )
                ],
                notes="missing job is 999999",
            )
        ],
    )

    bound = AgentEvaluationDatasetBinder.bind(
        dataset=dataset,
        manifest=_manifest(),
    )

    case = bound.cases[0]
    assert case.query == "查看文档 101；不要访问 202。"
    assert case.notes == "missing job is 303"
    assert case.expected_tool_calls[0].expected_arguments == {
        "document_id": 101
    }
    assert bound.fixture_placeholders["primary_document_id"] == 1001
    assert bound.fixture_bindings == {
        "primary_document_id": 101,
        "cross_user_document_id": 202,
        "missing_processing_job_id": 303,
    }
    AgentEvaluationDatasetBinder.ensure_live_ready(bound)


def test_dataset_binder_rejects_missing_binding() -> None:
    dataset = AgentEvaluationDataset(
        schema_version="1.0",
        dataset_id="dataset",
        dataset_version="1.0.0",
        description="fixture binding",
        fixture_placeholders={"primary_document_id": 1001},
        cases=[
            AgentEvaluationCase(
                case_id="document",
                query="文档 1001",
                category=AgentEvaluationCaseCategory.ONE_TOOL,
                expected_behavior="读取文档",
                allowed_tools=["get_document"],
                expected_answerable=True,
                expected_tool_calls=[
                    AgentExpectedToolCall(
                        tool_name="get_document",
                        expected_arguments={"document_id": 1001},
                    )
                ],
            )
        ],
    )
    manifest = _manifest(bindings={"cross_user_document_id": 202})

    with pytest.raises(ValueError, match="incomplete"):
        AgentEvaluationDatasetBinder.bind(
            dataset=dataset,
            manifest=manifest,
        )


def test_live_ready_rejects_unbound_template_dataset() -> None:
    dataset = AgentEvaluationCaseLoader.load_dataset(AGENT_CASES_PATH)

    with pytest.raises(ValueError, match="requires fixture binding"):
        AgentEvaluationDatasetBinder.ensure_live_ready(dataset)


def test_real_agent_dataset_declares_all_environment_placeholders() -> None:
    dataset = AgentEvaluationCaseLoader.load_dataset(AGENT_CASES_PATH)

    assert dataset.dataset_version == "1.1.0"
    assert dataset.fixture_placeholders == {
        "primary_document_id": 1001,
        "cross_user_document_id": 999998,
        "missing_processing_job_id": 999999,
    }
    assert dataset.fixture_bindings == {}


def test_manifest_round_trip_loader(tmp_path: Path) -> None:
    path = tmp_path / "fixture.json"
    manifest = _manifest()
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    loaded = AgentEvaluationDatasetBinder.load_manifest(path)

    assert loaded == manifest


def test_fixture_prepare_rejects_production_environment() -> None:
    with pytest.raises(RuntimeError, match="cannot be prepared in production"):
        _ensure_safe_environment("production")

    _ensure_safe_environment("development")
    _ensure_safe_environment("test")
