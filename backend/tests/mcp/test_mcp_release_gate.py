from datetime import datetime, timezone
from pathlib import Path

from app.schemas.agent_runtime_comparison import AgentRuntimeComparisonReport
from app.schemas.mcp_release import MCPLiveVerificationReport
from app.services.evaluation.mcp_release_readiness_service import (
    MCPReleaseReadinessService,
)


def _live_report(decision: str = "pass") -> MCPLiveVerificationReport:
    return MCPLiveVerificationReport(
        generated_at=datetime.now(timezone.utc),
        verifier_version="1.0.0",
        sdk_version="1.28.0",
        transport="stdio",
        server_id="release_probe",
        remote_tool_name="echo",
        exposed_tool_name="mcp__release_probe__echo",
        tool_contract_version="mcp-v1:test",
        mcp_toolset_version="toolset-v2:test",
        discovery_succeeded=decision == "pass",
        dispatch_succeeded=decision == "pass",
        structured_output_succeeded=decision == "pass",
        result_echo="v2.2-release-probe" if decision == "pass" else None,
        decision=decision,
        failure_reason=None if decision == "pass" else "probe failed",
    )


def _comparison() -> AgentRuntimeComparisonReport:
    path = Path("evaluation/reports/agent_runtime_comparison_v1.json")
    report = AgentRuntimeComparisonReport.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    return report.model_copy(
        update={
            "toolset_version": "toolset-v2:test",
            "tool_names": [
                "search_knowledge",
                "mcp__release_probe__echo",
            ],
        }
    )


def test_release_gate_passes_with_live_probe_and_fresh_agent_evidence():
    result = MCPReleaseReadinessService().evaluate(
        live_report=_live_report(),
        comparison=_comparison(),
    )

    assert result.decision == "pass"
    assert result.release_ready is True
    assert result.reasons == []


def test_release_gate_is_inconclusive_when_agent_eval_did_not_include_mcp_tool():
    comparison = _comparison().model_copy(
        update={"tool_names": ["search_knowledge"]}
    )

    result = MCPReleaseReadinessService().evaluate(
        live_report=_live_report(),
        comparison=comparison,
    )

    assert result.decision == "inconclusive"
    assert result.release_ready is False
    assert "mcp_tool_missing_from_agent_eval_toolset" in result.reasons
