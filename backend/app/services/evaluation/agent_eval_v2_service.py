from __future__ import annotations

from app.schemas.agent_evaluation import (
    AgentEvaluationDataset,
    AgentEvaluationDatasetReference,
    AgentEvaluationObservationSet,
)
from app.schemas.evaluation_v2 import (
    EvaluationCaseResultV2,
    EvaluationMetric,
    EvaluationMetricDirection,
    EvaluationMetricSource,
    EvaluationRunV2,
)
from app.services.evaluation.agent_evaluator import AgentEvaluator
from app.services.evaluation.eval_v2_adapter import EvaluationV2Adapter


class AgentEvaluationV2Service:
    """
    在不破坏 Agent Eval 1.0 报告的前提下补充 Eval 2.0 指标。

    当前新增的确定性能力：Tool Sequence Accuracy。
    Groundedness 继续复用已有 LLM-as-Judge 证据；Recovery 由 Stateful
    Eval 通过统一 Adapter 进入 Eval 2.0。
    """

    EVALUATOR_VERSION = "2.0.0"

    def __init__(self, base_evaluator: AgentEvaluator | None = None) -> None:
        self.base_evaluator = base_evaluator or AgentEvaluator()

    def evaluate(
        self,
        *,
        dataset: AgentEvaluationDataset,
        dataset_reference: AgentEvaluationDatasetReference,
        observations: AgentEvaluationObservationSet,
    ) -> EvaluationRunV2:
        base_report = self.base_evaluator.evaluate(
            dataset=dataset,
            dataset_reference=dataset_reference,
            observations=observations,
        )
        base_run = EvaluationV2Adapter.from_agent_report(base_report)

        observation_by_id = {
            observation.case_id: observation
            for observation in observations.observations
        }
        case_by_id = {case.case_id: case for case in dataset.cases}
        sequence_pass_by_id = {
            case_id: self._sequence_pass(
                expected=[call.tool_name for call in case.expected_tool_calls],
                actual=[
                    call.tool_name
                    for call in observation_by_id[case_id].tool_calls
                ],
            )
            for case_id, case in case_by_id.items()
        }

        base_case_by_id = {case.case_id: case for case in base_run.cases}
        v2_cases = tuple(
            self._upgrade_case(
                base_case_by_id[case.case_id],
                sequence_pass=sequence_pass_by_id[case.case_id],
            )
            for case in dataset.cases
        )
        v2_task_success_rate = sum(
            case.passed is True for case in v2_cases
        ) / len(v2_cases)
        sequence_accuracy = sum(sequence_pass_by_id.values()) / len(
            sequence_pass_by_id
        )

        summary_metrics = [
            metric
            for metric in base_run.summary_metrics
            if metric.metric_id != "task_success_rate"
        ]
        summary_metrics.insert(
            0,
            EvaluationMetric(
                metric_id="task_success_rate",
                value=v2_task_success_rate,
                source=EvaluationMetricSource.DETERMINISTIC,
                direction=EvaluationMetricDirection.HIGHER_IS_BETTER,
            ),
        )
        summary_metrics.append(
            EvaluationMetric(
                metric_id="tool_sequence_accuracy",
                value=sequence_accuracy,
                source=EvaluationMetricSource.DETERMINISTIC,
                direction=EvaluationMetricDirection.HIGHER_IS_BETTER,
            )
        )

        return base_run.model_copy(
            update={
                "version": base_run.version.model_copy(
                    update={"evaluator_version": self.EVALUATOR_VERSION}
                ),
                "summary_metrics": tuple(summary_metrics),
                "cases": v2_cases,
            }
        )

    @staticmethod
    def _sequence_pass(*, expected: list[str], actual: list[str]) -> bool:
        return actual == expected

    @staticmethod
    def _upgrade_case(
        base_case: EvaluationCaseResultV2,
        *,
        sequence_pass: bool,
    ) -> EvaluationCaseResultV2:
        base_pass = base_case.passed is not False
        return base_case.model_copy(
            update={
                "passed": base_pass and sequence_pass,
                "metrics": base_case.metrics
                + (
                    EvaluationMetric(
                        metric_id="tool_sequence_pass",
                        value=1 if sequence_pass else 0,
                        source=EvaluationMetricSource.DETERMINISTIC,
                        direction=(
                            EvaluationMetricDirection.HIGHER_IS_BETTER
                        ),
                    ),
                ),
            }
        )
