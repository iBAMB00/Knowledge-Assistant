from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.api.agent as agent_api
from app.api.dependencies.agent import get_agent_runtime_diagnostics_service
from app.api.dependencies.auth import get_current_user
from app.constants.agent_evaluation_runtime import (
    NATIVE_LIVE_EVALUATION_RUNNER_VERSION,
)
from app.models.database.user import User
from app.schemas.agent_evaluation import AgentEvaluationDatasetReference
from app.schemas.agent_runtime_comparison import (
    AgentRuntimeComparisonReport,
    AgentRuntimeComparisonSummary,
)
from app.services.agent_runtime_diagnostics_service import (
    AgentRuntimeDiagnosticsService,
)
from app.services.evaluation.agent_evaluator import AgentEvaluator
from app.services.evaluation.agent_live_evaluation_runner import (
    AgentLiveEvaluationRunner,
)
from scripts import check_agent_framework_readiness


def _comparison_report(
    *,
    decision: str = "pass",
    baseline_runner_version: str = NATIVE_LIVE_EVALUATION_RUNNER_VERSION,
    candidate_runner_version: str = "langchain-v1:1.4.0",
    evaluator_version: str = AgentEvaluator.EVALUATOR_VERSION,
) -> AgentRuntimeComparisonReport:
    return AgentRuntimeComparisonReport(
        generated_at=datetime.now(timezone.utc),
        dataset=AgentEvaluationDatasetReference(
            schema_version="1.0",
            dataset_id="knowledge-assistant-agent-eval",
            dataset_version="1.5.0",
            source_path="evaluation/generated/agent_cases.bound.json",
            source_sha256="a" * 64,
            total_cases=9,
        ),
        evaluator_version=evaluator_version,
        baseline_runner_version=baseline_runner_version,
        candidate_runner_version=candidate_runner_version,
        summary=AgentRuntimeComparisonSummary(
            decision=decision,
            deterministic_gate_passed=decision != "fail",
            groundedness_gate_status=(
                "inconclusive" if decision == "inconclusive" else decision
            ),
            failed_metrics=(
                ["tool_selection_accuracy"] if decision == "fail" else []
            ),
            inconclusive_metrics=(
                ["grounded_answer_rate"]
                if decision == "inconclusive"
                else []
            ),
            regression_case_ids=(
                ["multi_tool_document_status"] if decision == "fail" else []
            ),
            improvement_case_ids=[],
            task_success_rate_delta=0.0,
            average_tool_calls_delta=0.0,
            average_latency_ms_delta=0.0,
            average_latency_ratio=1.0,
        ),
        metric_checks=[],
        case_comparisons=[],
    )


def test_runtime_status_is_lightweight_and_reflects_feature_gate() -> None:
    service = AgentRuntimeDiagnosticsService(
        langchain_candidate_enabled=False
    )

    status = service.get_runtime_status()

    assert status.default_runtime.value == "native"
    assert [item.runtime.value for item in status.runtimes] == [
        "native",
        "langchain",
    ]
    native, candidate = status.runtimes
    assert native.role == "baseline"
    assert native.enabled is True
    assert native.supports_sync is True
    assert native.supports_stream is True
    assert native.implementation_version == "2.0.0"
    assert candidate.role == "candidate"
    assert candidate.enabled is False
    assert candidate.supports_sync is True
    assert candidate.supports_stream is True
    assert candidate.implementation_version == "langchain-v1:1.4.0"


def test_runtime_status_enables_candidate_without_changing_default() -> None:
    service = AgentRuntimeDiagnosticsService(
        langchain_candidate_enabled=True
    )

    status = service.get_runtime_status()

    assert status.default_runtime.value == "native"
    candidate = next(
        item for item in status.runtimes if item.runtime.value == "langchain"
    )
    assert candidate.enabled is True


def test_release_gate_passes_only_with_current_comparison_evidence() -> None:
    service = AgentRuntimeDiagnosticsService(
        langchain_candidate_enabled=True
    )

    result = service.evaluate_framework_release(_comparison_report())

    assert result.decision == "pass"
    assert result.release_ready is True
    assert result.reasons == []
    assert result.candidate_runner_version == "langchain-v1:1.4.0"
    assert result.expected_candidate_runner_version == "langchain-v1:1.4.0"
    assert result.expected_evaluator_version == AgentEvaluator.EVALUATOR_VERSION


def test_release_gate_marks_old_candidate_pass_as_inconclusive() -> None:
    service = AgentRuntimeDiagnosticsService(
        langchain_candidate_enabled=True
    )

    result = service.evaluate_framework_release(
        _comparison_report(candidate_runner_version="langchain-v1:1.3.0")
    )

    assert result.decision == "inconclusive"
    assert result.release_ready is False
    assert result.comparison_decision == "pass"
    assert result.reasons == ["candidate_comparison_evidence_is_stale"]


def test_release_gate_preserves_real_comparison_failure() -> None:
    service = AgentRuntimeDiagnosticsService(
        langchain_candidate_enabled=False
    )

    result = service.evaluate_framework_release(
        _comparison_report(decision="fail")
    )

    assert result.decision == "fail"
    assert result.release_ready is False
    assert "runtime_comparison_failed" in result.reasons
    # Feature gate 是部署策略，不会把质量 FAIL 改写成其他状态。
    assert result.candidate_feature_gate_enabled is False


def test_native_eval_runner_version_uses_shared_constant() -> None:
    assert (
        AgentLiveEvaluationRunner.RUNNER_VERSION
        == NATIVE_LIVE_EVALUATION_RUNNER_VERSION
    )


def test_agent_runtimes_endpoint_is_authenticated_and_does_not_expose_model() -> None:
    service = AgentRuntimeDiagnosticsService(
        langchain_candidate_enabled=True
    )
    app = FastAPI()
    app.include_router(agent_api.router)
    app.dependency_overrides[get_current_user] = lambda: User(
        id=9,
        email="runtime-diagnostics@example.com",
        password_hash="hash",
        role="user",
        is_active=True,
    )
    app.dependency_overrides[
        get_agent_runtime_diagnostics_service
    ] = lambda: service

    with TestClient(app) as client:
        response = client.get("/agent/runtimes")

    assert response.status_code == 200
    payload = response.json()
    assert payload["default_runtime"] == "native"
    assert payload["runtimes"][1]["runtime"] == "langchain"
    assert payload["runtimes"][1]["enabled"] is True
    assert payload["runtimes"][1]["implementation_version"] == (
        "langchain-v1:1.4.0"
    )
    rendered = json.dumps(payload)
    assert "model_name" not in rendered
    assert "api_key" not in rendered


def test_readiness_script_writes_inconclusive_for_stale_candidate_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    comparison_path = tmp_path / "comparison.json"
    output_path = tmp_path / "release_gate.json"
    comparison_path.write_text(
        _comparison_report(
            candidate_runner_version="langchain-v1:1.3.0"
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        check_agent_framework_readiness,
        "get_settings",
        lambda: SimpleNamespace(agent_langchain_candidate_enabled=True),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_agent_framework_readiness",
            "--comparison",
            str(comparison_path),
            "--output",
            str(output_path),
            "--fail-on-not-ready",
        ],
    )

    assert check_agent_framework_readiness.main() == 3
    rendered = json.loads(capsys.readouterr().out)
    assert rendered["decision"] == "inconclusive"
    assert rendered["candidate_runner_version"] == "langchain-v1:1.3.0"
    assert rendered["expected_candidate_runner_version"] == (
        "langchain-v1:1.4.0"
    )
    assert rendered["reasons"] == [
        "candidate_comparison_evidence_is_stale"
    ]
    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    assert persisted["release_ready"] is False


def test_release_gate_treats_feature_gate_as_deployment_policy_only() -> None:
    service = AgentRuntimeDiagnosticsService(
        langchain_candidate_enabled=False
    )

    result = service.evaluate_framework_release(_comparison_report())

    assert result.decision == "pass"
    assert result.release_ready is True
    assert result.candidate_feature_gate_enabled is False


def test_agent_runtimes_endpoint_requires_bearer_authentication() -> None:
    from app.core.database import get_db

    app = FastAPI()
    app.include_router(agent_api.router)

    def override_get_db():
        yield None

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as client:
        response = client.get("/agent/runtimes")

    assert response.status_code == 401
    assert response.json() == {
        "detail": "invalid or missing access token"
    }
