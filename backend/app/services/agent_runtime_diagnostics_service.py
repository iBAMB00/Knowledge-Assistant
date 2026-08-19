from __future__ import annotations

from app.agent.frameworks.langchain.runner import LangChainSingleAgentRunner
from app.agent.version_snapshot import AGENT_RUNTIME_VERSION
from app.constants.agent_runtime import AgentRuntime
from app.schemas.agent_runtime_comparison import AgentRuntimeComparisonReport
from app.schemas.agent_runtime_status import (
    AgentFrameworkReleaseGateResult,
    AgentRuntimeCapabilityResponse,
    AgentRuntimeStatusResponse,
)
from app.services.evaluation.agent_evaluator import AgentEvaluator
from app.constants.agent_evaluation_runtime import (
    NATIVE_LIVE_EVALUATION_RUNNER_VERSION,
)


class AgentRuntimeDiagnosticsService:
    """
    汇总 Agent Runtime 能力，并校验 v2.1 Framework Candidate 的发布证据。

    本服务只读取常量、配置开关和离线 Comparison Report，不构造模型、
    Tool、数据库 Session 或 Framework Graph，因此可安全用于轻量诊断。
    """

    DEFAULT_RUNTIME = AgentRuntime.NATIVE

    def __init__(self, *, langchain_candidate_enabled: bool) -> None:
        self._langchain_candidate_enabled = langchain_candidate_enabled

    def get_runtime_status(self) -> AgentRuntimeStatusResponse:
        """返回 Native Baseline 与 LangChain Candidate 的部署能力摘要。"""

        return AgentRuntimeStatusResponse(
            default_runtime=self.DEFAULT_RUNTIME,
            runtimes=[
                AgentRuntimeCapabilityResponse(
                    runtime=AgentRuntime.NATIVE,
                    role="baseline",
                    enabled=True,
                    supports_sync=True,
                    supports_stream=True,
                    implementation_version=AGENT_RUNTIME_VERSION,
                ),
                AgentRuntimeCapabilityResponse(
                    runtime=AgentRuntime.LANGCHAIN,
                    role="candidate",
                    enabled=self._langchain_candidate_enabled,
                    supports_sync=True,
                    supports_stream=True,
                    implementation_version=self.expected_candidate_version,
                ),
            ],
        )

    def evaluate_framework_release(
        self,
        comparison: AgentRuntimeComparisonReport,
    ) -> AgentFrameworkReleaseGateResult:
        """
        判断当前代码版本是否拥有同版本、同 Evaluator 的有效 A6 PASS 证据。

        Comparison FAIL 表示真实回归；Comparison INCONCLUSIVE 或版本不匹配
        表示证据不足/陈旧，不应误判为当前代码可发布。
        Candidate feature gate 仅记录部署状态，不作为代码发布质量门禁。
        """

        reasons: list[str] = []
        comparison_decision = comparison.summary.decision

        if comparison_decision == "fail":
            reasons.append("runtime_comparison_failed")
        elif comparison_decision == "inconclusive":
            reasons.append("runtime_comparison_inconclusive")

        expected_baseline = NATIVE_LIVE_EVALUATION_RUNNER_VERSION
        expected_candidate = self.expected_candidate_version
        expected_evaluator = AgentEvaluator.EVALUATOR_VERSION

        if comparison.baseline_runner_version != expected_baseline:
            reasons.append("baseline_comparison_evidence_is_stale")
        if comparison.candidate_runner_version != expected_candidate:
            reasons.append("candidate_comparison_evidence_is_stale")
        if comparison.evaluator_version != expected_evaluator:
            reasons.append("evaluator_comparison_evidence_is_stale")

        if comparison_decision == "fail":
            decision = "fail"
        elif reasons:
            decision = "inconclusive"
        else:
            decision = "pass"

        return AgentFrameworkReleaseGateResult(
            decision=decision,
            release_ready=decision == "pass",
            comparison_decision=comparison_decision,
            dataset_id=comparison.dataset.dataset_id,
            dataset_version=comparison.dataset.dataset_version,
            evaluator_version=comparison.evaluator_version,
            expected_evaluator_version=expected_evaluator,
            baseline_runner_version=comparison.baseline_runner_version,
            expected_baseline_runner_version=expected_baseline,
            candidate_runner_version=comparison.candidate_runner_version,
            expected_candidate_runner_version=expected_candidate,
            candidate_feature_gate_enabled=(
                self._langchain_candidate_enabled
            ),
            reasons=reasons,
        )

    @property
    def expected_candidate_version(self) -> str:
        """返回当前代码对应的 LangChain Candidate 运行时版本。"""

        return f"langchain-v1:{LangChainSingleAgentRunner.RUNNER_VERSION}"
