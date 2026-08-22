from __future__ import annotations

from datetime import datetime, timezone

from app.schemas.agent_evaluation import (
    AgentEvaluationCaseResult,
    AgentEvaluationObservation,
    AgentEvaluationObservationSet,
    AgentEvaluationReport,
)
from app.schemas.agent_runtime_comparison import (
    AgentRuntimeCaseComparison,
    AgentRuntimeComparisonReport,
    AgentRuntimeComparisonSummary,
    AgentRuntimeMetricCheck,
    GroundednessGateStatus,
)


class AgentRuntimeComparisonService:
    """
    Native Baseline 与 Framework Candidate 的 Runtime 回归对比服务。

    设计原则：
    - Tool / Policy / Evidence / Citation 等确定性指标参与硬门禁。
    - Groundedness 只有在双方 Judge coverage 完整时参与硬判定。
    - Task Success 由于当前会被 Groundedness Judge 可用性影响，仅展示差值。
    - Latency / Tool Call 数只做观测，不在 v2.1 收口阶段作为硬失败条件。
    """

    FLOAT_EPSILON = 1e-12

    def compare(
        self,
        *,
        baseline_report: AgentEvaluationReport,
        baseline_observations: AgentEvaluationObservationSet,
        candidate_report: AgentEvaluationReport,
        candidate_observations: AgentEvaluationObservationSet,
    ) -> AgentRuntimeComparisonReport:
        self._validate_compatibility(
            baseline_report=baseline_report,
            baseline_observations=baseline_observations,
            candidate_report=candidate_report,
            candidate_observations=candidate_observations,
        )

        metric_checks = self._build_metric_checks(
            baseline_report=baseline_report,
            candidate_report=candidate_report,
        )
        case_comparisons = self._build_case_comparisons(
            baseline_report=baseline_report,
            baseline_observations=baseline_observations,
            candidate_report=candidate_report,
            candidate_observations=candidate_observations,
        )

        failed_metrics = [
            check.metric
            for check in metric_checks
            if check.status == "fail"
            and check.metric != "grounded_answer_rate"
        ]
        inconclusive_metrics = [
            check.metric
            for check in metric_checks
            if check.status == "inconclusive"
        ]
        deterministic_inconclusive_metrics = [
            metric
            for metric in inconclusive_metrics
            if metric != "grounded_answer_rate"
        ]
        regression_case_ids = [
            comparison.case_id
            for comparison in case_comparisons
            if comparison.regression_reasons
        ]
        improvement_case_ids = [
            comparison.case_id
            for comparison in case_comparisons
            if comparison.improvement_reasons
        ]

        deterministic_gate_passed = (
            not failed_metrics
            and not deterministic_inconclusive_metrics
            and not regression_case_ids
        )
        groundedness_gate_status = self._groundedness_gate_status(
            metric_checks
        )

        if (
            failed_metrics
            or regression_case_ids
            or groundedness_gate_status == "fail"
        ):
            decision = "fail"
        elif (
            deterministic_inconclusive_metrics
            or groundedness_gate_status == "inconclusive"
        ):
            decision = "inconclusive"
        else:
            decision = "pass"

        baseline_summary = baseline_report.summary
        candidate_summary = candidate_report.summary
        latency_delta = (
            candidate_summary.average_latency_ms
            - baseline_summary.average_latency_ms
        )
        latency_ratio = (
            candidate_summary.average_latency_ms
            / baseline_summary.average_latency_ms
            if baseline_summary.average_latency_ms > 0
            else None
        )

        return AgentRuntimeComparisonReport(
            generated_at=datetime.now(timezone.utc),
            dataset=baseline_report.dataset,
            evaluator_version=baseline_report.evaluator_version,
            baseline_runner_version=self._require_runner_version(
                baseline_observations,
                label="baseline",
            ),
            candidate_runner_version=self._require_runner_version(
                candidate_observations,
                label="candidate",
            ),
            toolset_version=self._shared_toolset_version(
                baseline_observations,
                candidate_observations,
            ),
            tool_names=self._shared_tool_names(
                baseline_observations,
                candidate_observations,
            ),
            summary=AgentRuntimeComparisonSummary(
                decision=decision,
                deterministic_gate_passed=deterministic_gate_passed,
                groundedness_gate_status=groundedness_gate_status,
                failed_metrics=failed_metrics,
                inconclusive_metrics=inconclusive_metrics,
                regression_case_ids=regression_case_ids,
                improvement_case_ids=improvement_case_ids,
                task_success_rate_delta=(
                    candidate_summary.task_success_rate
                    - baseline_summary.task_success_rate
                ),
                average_tool_calls_delta=(
                    candidate_summary.average_tool_calls
                    - baseline_summary.average_tool_calls
                ),
                average_latency_ms_delta=latency_delta,
                average_latency_ratio=latency_ratio,
            ),
            metric_checks=metric_checks,
            case_comparisons=case_comparisons,
        )

    def _validate_compatibility(
        self,
        *,
        baseline_report: AgentEvaluationReport,
        baseline_observations: AgentEvaluationObservationSet,
        candidate_report: AgentEvaluationReport,
        candidate_observations: AgentEvaluationObservationSet,
    ) -> None:
        if baseline_report.dataset != candidate_report.dataset:
            raise ValueError(
                "baseline and candidate must use the same dataset snapshot"
            )
        if baseline_report.evaluator_version != candidate_report.evaluator_version:
            raise ValueError(
                "baseline and candidate must use the same evaluator version"
            )

        self._validate_observation_identity(
            report=baseline_report,
            observations=baseline_observations,
            label="baseline",
        )
        self._validate_observation_identity(
            report=candidate_report,
            observations=candidate_observations,
            label="candidate",
        )

        baseline_report_ids = {case.case_id for case in baseline_report.cases}
        candidate_report_ids = {case.case_id for case in candidate_report.cases}
        baseline_observation_ids = {
            observation.case_id
            for observation in baseline_observations.observations
        }
        candidate_observation_ids = {
            observation.case_id
            for observation in candidate_observations.observations
        }
        if not (
            baseline_report_ids
            == candidate_report_ids
            == baseline_observation_ids
            == candidate_observation_ids
        ):
            raise ValueError(
                "baseline and candidate report/observation case IDs must match"
            )

        baseline_category_by_id = {
            case.case_id: case.category
            for case in baseline_report.cases
        }
        candidate_category_by_id = {
            case.case_id: case.category
            for case in candidate_report.cases
        }
        if baseline_category_by_id != candidate_category_by_id:
            raise ValueError(
                "baseline and candidate case categories must match"
            )

        self._require_runner_version(baseline_observations, label="baseline")
        self._require_runner_version(candidate_observations, label="candidate")


    @staticmethod
    def _shared_toolset_version(
        baseline: AgentEvaluationObservationSet,
        candidate: AgentEvaluationObservationSet,
    ) -> str | None:
        if baseline.toolset_version is None and candidate.toolset_version is None:
            return None
        if baseline.toolset_version is None or candidate.toolset_version is None:
            raise ValueError("baseline/candidate toolset_version metadata must both exist")
        if baseline.toolset_version != candidate.toolset_version:
            raise ValueError("baseline and candidate must use the same toolset_version")
        return baseline.toolset_version

    @staticmethod
    def _shared_tool_names(
        baseline: AgentEvaluationObservationSet,
        candidate: AgentEvaluationObservationSet,
    ) -> list[str]:
        if not baseline.tool_names and not candidate.tool_names:
            return []
        if not baseline.tool_names or not candidate.tool_names:
            raise ValueError("baseline/candidate tool_names metadata must both exist")
        baseline_names = sorted(baseline.tool_names)
        candidate_names = sorted(candidate.tool_names)
        if baseline_names != candidate_names:
            raise ValueError("baseline and candidate must expose the same Agent Toolset")
        return baseline_names

    @staticmethod
    def _validate_observation_identity(
        *,
        report: AgentEvaluationReport,
        observations: AgentEvaluationObservationSet,
        label: str,
    ) -> None:
        if observations.dataset_id != report.dataset.dataset_id:
            raise ValueError(
                f"{label} observation dataset_id does not match report"
            )
        if observations.dataset_version != report.dataset.dataset_version:
            raise ValueError(
                f"{label} observation dataset_version does not match report"
            )

    @staticmethod
    def _require_runner_version(
        observations: AgentEvaluationObservationSet,
        *,
        label: str,
    ) -> str:
        runner_version = observations.runner_version
        if runner_version is None:
            raise ValueError(f"{label} observations require runner_version")
        return runner_version

    def _build_metric_checks(
        self,
        *,
        baseline_report: AgentEvaluationReport,
        candidate_report: AgentEvaluationReport,
    ) -> list[AgentRuntimeMetricCheck]:
        baseline = baseline_report.summary
        candidate = candidate_report.summary

        checks = [
            self._higher_is_better(
                "tool_selection_accuracy",
                baseline.tool_selection_accuracy,
                candidate.tool_selection_accuracy,
            ),
            self._higher_is_better(
                "tool_execution_accuracy",
                baseline.tool_execution_accuracy,
                candidate.tool_execution_accuracy,
            ),
            self._higher_is_better(
                "tool_argument_accuracy",
                baseline.tool_argument_accuracy,
                candidate.tool_argument_accuracy,
            ),
            self._lower_is_better(
                "unnecessary_tool_call_rate",
                baseline.unnecessary_tool_call_rate,
                candidate.unnecessary_tool_call_rate,
            ),
            self._lower_is_better(
                "tool_policy_violation_count",
                float(baseline.tool_policy_violation_count),
                float(candidate.tool_policy_violation_count),
            ),
            self._higher_is_better(
                "required_evidence_success_rate",
                baseline.required_evidence_success_rate,
                candidate.required_evidence_success_rate,
            ),
            self._higher_is_better(
                "required_citation_success_rate",
                baseline.required_citation_success_rate,
                candidate.required_citation_success_rate,
            ),
            self._higher_is_better(
                "citation_correctness",
                baseline.citation_correctness,
                candidate.citation_correctness,
            ),
        ]

        checks.extend(
            self._groundedness_checks(
                baseline_report=baseline_report,
                candidate_report=candidate_report,
            )
        )
        checks.extend(
            [
                self._informational(
                    "task_success_rate",
                    baseline.task_success_rate,
                    candidate.task_success_rate,
                ),
                self._informational(
                    "average_tool_calls",
                    baseline.average_tool_calls,
                    candidate.average_tool_calls,
                ),
                self._informational(
                    "average_latency_ms",
                    baseline.average_latency_ms,
                    candidate.average_latency_ms,
                ),
            ]
        )
        return checks

    def _groundedness_checks(
        self,
        *,
        baseline_report: AgentEvaluationReport,
        candidate_report: AgentEvaluationReport,
    ) -> list[AgentRuntimeMetricCheck]:
        baseline = baseline_report.summary
        candidate = candidate_report.summary
        baseline_coverage = baseline.groundedness_coverage
        candidate_coverage = candidate.groundedness_coverage

        coverage_check = self._informational(
            "groundedness_coverage",
            baseline_coverage,
            candidate_coverage,
        )

        if baseline_coverage is None and candidate_coverage is None:
            groundedness_check = AgentRuntimeMetricCheck(
                metric="grounded_answer_rate",
                direction="higher_is_better",
                baseline_value=baseline.grounded_answer_rate,
                candidate_value=candidate.grounded_answer_rate,
                delta=self._delta(
                    baseline.grounded_answer_rate,
                    candidate.grounded_answer_rate,
                ),
                status="informational",
                reason="groundedness_not_applicable",
            )
            return [coverage_check, groundedness_check]

        if (
            baseline_coverage is None
            or candidate_coverage is None
            or baseline_coverage < 1.0 - self.FLOAT_EPSILON
            or candidate_coverage < 1.0 - self.FLOAT_EPSILON
        ):
            groundedness_check = AgentRuntimeMetricCheck(
                metric="grounded_answer_rate",
                direction="higher_is_better",
                baseline_value=baseline.grounded_answer_rate,
                candidate_value=candidate.grounded_answer_rate,
                delta=self._delta(
                    baseline.grounded_answer_rate,
                    candidate.grounded_answer_rate,
                ),
                status="inconclusive",
                reason="groundedness_judge_coverage_incomplete",
            )
            return [coverage_check, groundedness_check]

        groundedness_check = self._higher_is_better(
            "grounded_answer_rate",
            baseline.grounded_answer_rate,
            candidate.grounded_answer_rate,
        )
        return [coverage_check, groundedness_check]

    def _build_case_comparisons(
        self,
        *,
        baseline_report: AgentEvaluationReport,
        baseline_observations: AgentEvaluationObservationSet,
        candidate_report: AgentEvaluationReport,
        candidate_observations: AgentEvaluationObservationSet,
    ) -> list[AgentRuntimeCaseComparison]:
        baseline_cases = {
            case.case_id: case
            for case in baseline_report.cases
        }
        candidate_cases = {
            case.case_id: case
            for case in candidate_report.cases
        }
        baseline_obs = {
            observation.case_id: observation
            for observation in baseline_observations.observations
        }
        candidate_obs = {
            observation.case_id: observation
            for observation in candidate_observations.observations
        }

        comparisons: list[AgentRuntimeCaseComparison] = []
        for baseline_case in baseline_report.cases:
            case_id = baseline_case.case_id
            candidate_case = candidate_cases[case_id]
            baseline_observation = baseline_obs[case_id]
            candidate_observation = candidate_obs[case_id]

            regressions = self._case_regression_reasons(
                baseline_case=baseline_cases[case_id],
                baseline_observation=baseline_observation,
                candidate_case=candidate_case,
                candidate_observation=candidate_observation,
            )
            improvements = self._case_improvement_reasons(
                baseline_case=baseline_cases[case_id],
                baseline_observation=baseline_observation,
                candidate_case=candidate_case,
                candidate_observation=candidate_observation,
            )
            comparisons.append(
                AgentRuntimeCaseComparison(
                    case_id=case_id,
                    category=baseline_case.category,
                    regression_reasons=regressions,
                    improvement_reasons=improvements,
                    baseline_run_succeeded=baseline_observation.run_succeeded,
                    candidate_run_succeeded=candidate_observation.run_succeeded,
                    baseline_task_success=baseline_case.task_success,
                    candidate_task_success=candidate_case.task_success,
                    baseline_tool_call_count=baseline_case.tool_call_count,
                    candidate_tool_call_count=candidate_case.tool_call_count,
                    baseline_latency_ms=baseline_case.latency_ms,
                    candidate_latency_ms=candidate_case.latency_ms,
                )
            )
        return comparisons

    def _case_regression_reasons(
        self,
        *,
        baseline_case: AgentEvaluationCaseResult,
        baseline_observation: AgentEvaluationObservation,
        candidate_case: AgentEvaluationCaseResult,
        candidate_observation: AgentEvaluationObservation,
    ) -> list[str]:
        reasons: list[str] = []

        if baseline_observation.run_succeeded and not candidate_observation.run_succeeded:
            reasons.append("run_succeeded_lost")
        if baseline_case.tool_selection_pass and not candidate_case.tool_selection_pass:
            reasons.append("tool_selection_regressed")
        if baseline_case.tool_execution_pass is True and candidate_case.tool_execution_pass is not True:
            reasons.append("tool_execution_regressed")
        if self._optional_lower(
            candidate_case.tool_argument_accuracy,
            baseline_case.tool_argument_accuracy,
        ):
            reasons.append("tool_argument_accuracy_decreased")
        if (
            candidate_case.unnecessary_tool_call_rate
            > baseline_case.unnecessary_tool_call_rate + self.FLOAT_EPSILON
        ):
            reasons.append("unnecessary_tool_call_rate_increased")
        if (
            candidate_case.tool_policy_violation_count
            > baseline_case.tool_policy_violation_count
        ):
            reasons.append("tool_policy_violation_increased")
        if (
            baseline_case.retrieved_evidence_pass is True
            and candidate_case.retrieved_evidence_pass is not True
        ):
            reasons.append("required_evidence_regressed")
        if (
            baseline_case.citation_requirement_pass is True
            and candidate_case.citation_requirement_pass is not True
        ):
            reasons.append("required_citation_regressed")
        if self._optional_lower(
            candidate_case.citation_correctness,
            baseline_case.citation_correctness,
        ):
            reasons.append("citation_correctness_decreased")

        return reasons

    def _case_improvement_reasons(
        self,
        *,
        baseline_case: AgentEvaluationCaseResult,
        baseline_observation: AgentEvaluationObservation,
        candidate_case: AgentEvaluationCaseResult,
        candidate_observation: AgentEvaluationObservation,
    ) -> list[str]:
        reasons: list[str] = []

        if not baseline_observation.run_succeeded and candidate_observation.run_succeeded:
            reasons.append("run_succeeded_gained")
        if not baseline_case.tool_selection_pass and candidate_case.tool_selection_pass:
            reasons.append("tool_selection_improved")
        if baseline_case.tool_execution_pass is False and candidate_case.tool_execution_pass is True:
            reasons.append("tool_execution_improved")
        if self._optional_higher(
            candidate_case.tool_argument_accuracy,
            baseline_case.tool_argument_accuracy,
        ):
            reasons.append("tool_argument_accuracy_increased")
        if (
            candidate_case.unnecessary_tool_call_rate
            + self.FLOAT_EPSILON
            < baseline_case.unnecessary_tool_call_rate
        ):
            reasons.append("unnecessary_tool_call_rate_decreased")
        if (
            candidate_case.tool_policy_violation_count
            < baseline_case.tool_policy_violation_count
        ):
            reasons.append("tool_policy_violation_decreased")
        if (
            baseline_case.retrieved_evidence_pass is False
            and candidate_case.retrieved_evidence_pass is True
        ):
            reasons.append("required_evidence_improved")
        if (
            baseline_case.citation_requirement_pass is False
            and candidate_case.citation_requirement_pass is True
        ):
            reasons.append("required_citation_improved")
        if self._optional_higher(
            candidate_case.citation_correctness,
            baseline_case.citation_correctness,
        ):
            reasons.append("citation_correctness_increased")

        return reasons

    def _higher_is_better(
        self,
        metric: str,
        baseline_value: float | None,
        candidate_value: float | None,
    ) -> AgentRuntimeMetricCheck:
        if baseline_value is None and candidate_value is None:
            return AgentRuntimeMetricCheck(
                metric=metric,
                direction="higher_is_better",
                baseline_value=None,
                candidate_value=None,
                delta=None,
                status="informational",
                reason="metric_not_applicable",
            )
        if baseline_value is not None and candidate_value is None:
            return AgentRuntimeMetricCheck(
                metric=metric,
                direction="higher_is_better",
                baseline_value=baseline_value,
                candidate_value=None,
                delta=None,
                status="inconclusive",
                reason="candidate_metric_unavailable",
            )
        if baseline_value is None:
            return AgentRuntimeMetricCheck(
                metric=metric,
                direction="higher_is_better",
                baseline_value=None,
                candidate_value=candidate_value,
                delta=None,
                status="pass",
                reason="baseline_metric_not_applicable",
            )

        assert candidate_value is not None
        passed = candidate_value + self.FLOAT_EPSILON >= baseline_value
        return AgentRuntimeMetricCheck(
            metric=metric,
            direction="higher_is_better",
            baseline_value=baseline_value,
            candidate_value=candidate_value,
            delta=candidate_value - baseline_value,
            status="pass" if passed else "fail",
        )

    def _lower_is_better(
        self,
        metric: str,
        baseline_value: float | None,
        candidate_value: float | None,
    ) -> AgentRuntimeMetricCheck:
        if baseline_value is None and candidate_value is None:
            return AgentRuntimeMetricCheck(
                metric=metric,
                direction="lower_is_better",
                baseline_value=None,
                candidate_value=None,
                delta=None,
                status="informational",
                reason="metric_not_applicable",
            )
        if baseline_value is not None and candidate_value is None:
            return AgentRuntimeMetricCheck(
                metric=metric,
                direction="lower_is_better",
                baseline_value=baseline_value,
                candidate_value=None,
                delta=None,
                status="inconclusive",
                reason="candidate_metric_unavailable",
            )
        if baseline_value is None:
            return AgentRuntimeMetricCheck(
                metric=metric,
                direction="lower_is_better",
                baseline_value=None,
                candidate_value=candidate_value,
                delta=None,
                status="pass",
                reason="baseline_metric_not_applicable",
            )

        assert candidate_value is not None
        passed = candidate_value <= baseline_value + self.FLOAT_EPSILON
        return AgentRuntimeMetricCheck(
            metric=metric,
            direction="lower_is_better",
            baseline_value=baseline_value,
            candidate_value=candidate_value,
            delta=candidate_value - baseline_value,
            status="pass" if passed else "fail",
        )

    @staticmethod
    def _informational(
        metric: str,
        baseline_value: float | None,
        candidate_value: float | None,
    ) -> AgentRuntimeMetricCheck:
        return AgentRuntimeMetricCheck(
            metric=metric,
            direction="informational",
            baseline_value=baseline_value,
            candidate_value=candidate_value,
            delta=AgentRuntimeComparisonService._delta(
                baseline_value,
                candidate_value,
            ),
            status="informational",
        )

    @classmethod
    def _optional_lower(
        cls,
        candidate_value: float | None,
        baseline_value: float | None,
    ) -> bool:
        if baseline_value is None:
            return False
        if candidate_value is None:
            return True
        return candidate_value < baseline_value - cls.FLOAT_EPSILON

    @classmethod
    def _optional_higher(
        cls,
        candidate_value: float | None,
        baseline_value: float | None,
    ) -> bool:
        if candidate_value is None or baseline_value is None:
            return False
        return candidate_value > baseline_value + cls.FLOAT_EPSILON

    @staticmethod
    def _delta(
        baseline_value: float | None,
        candidate_value: float | None,
    ) -> float | None:
        if baseline_value is None or candidate_value is None:
            return None
        return candidate_value - baseline_value

    @staticmethod
    def _groundedness_gate_status(
        checks: list[AgentRuntimeMetricCheck],
    ) -> GroundednessGateStatus:
        groundedness_check = next(
            check
            for check in checks
            if check.metric == "grounded_answer_rate"
        )
        if groundedness_check.reason == "groundedness_not_applicable":
            return "not_applicable"
        if groundedness_check.status == "inconclusive":
            return "inconclusive"
        if groundedness_check.status == "fail":
            return "fail"
        return "pass"
