from __future__ import annotations

from datetime import datetime, timezone

from app.agent.frameworks.langchain.runner import LangChainSingleAgentRunner
from app.constants.agent_evaluation_runtime import (
    NATIVE_LIVE_EVALUATION_RUNNER_VERSION,
)
from app.schemas.agent_runtime_comparison import AgentRuntimeComparisonReport
from app.schemas.mcp_release import MCPReleaseGateResult, MCPLiveVerificationReport
from app.services.evaluation.agent_evaluator import AgentEvaluator


class MCPReleaseReadinessService:
    """组合真实 MCP 探针与同 Toolset Agent Eval，判断 v2.2 是否可发布。"""

    def evaluate(
        self,
        *,
        live_report: MCPLiveVerificationReport,
        comparison: AgentRuntimeComparisonReport,
    ) -> MCPReleaseGateResult:
        reasons: list[str] = []

        if live_report.decision == "fail":
            reasons.append("mcp_live_verification_failed")

        if comparison.summary.decision == "fail":
            reasons.append("runtime_comparison_failed")
        elif comparison.summary.decision == "inconclusive":
            reasons.append("runtime_comparison_inconclusive")

        expected_candidate = (
            f"langchain-v1:{LangChainSingleAgentRunner.RUNNER_VERSION}"
        )
        if (
            comparison.baseline_runner_version
            != NATIVE_LIVE_EVALUATION_RUNNER_VERSION
        ):
            reasons.append("baseline_evidence_is_stale")
        if comparison.candidate_runner_version != expected_candidate:
            reasons.append("candidate_evidence_is_stale")
        if comparison.evaluator_version != AgentEvaluator.EVALUATOR_VERSION:
            reasons.append("evaluator_evidence_is_stale")

        if comparison.toolset_version is None:
            reasons.append("runtime_comparison_missing_toolset_version")
        if not comparison.tool_names:
            reasons.append("runtime_comparison_missing_tool_names")
        elif live_report.exposed_tool_name not in comparison.tool_names:
            reasons.append("mcp_tool_missing_from_agent_eval_toolset")

        if live_report.decision == "fail" or comparison.summary.decision == "fail":
            decision = "fail"
        elif reasons:
            decision = "inconclusive"
        else:
            decision = "pass"

        return MCPReleaseGateResult(
            generated_at=datetime.now(timezone.utc),
            decision=decision,
            release_ready=decision == "pass",
            live_verification_decision=live_report.decision,
            runtime_comparison_decision=comparison.summary.decision,
            mcp_tool_name=live_report.exposed_tool_name,
            toolset_version=comparison.toolset_version,
            dataset_id=comparison.dataset.dataset_id,
            dataset_version=comparison.dataset.dataset_version,
            evaluator_version=comparison.evaluator_version,
            baseline_runner_version=comparison.baseline_runner_version,
            candidate_runner_version=comparison.candidate_runner_version,
            reasons=reasons,
        )
