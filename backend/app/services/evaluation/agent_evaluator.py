from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any

from app.schemas.agent_evaluation import (
    AgentEvaluationCase,
    AgentEvaluationCaseResult,
    AgentEvaluationDataset,
    AgentEvaluationDatasetReference,
    AgentEvaluationObservation,
    AgentEvaluationObservationSet,
    AgentEvaluationReport,
    AgentEvaluationSummary,
    AgentExpectedToolCall,
    AgentObservedToolCall,
)


class AgentEvaluator:
    """
    Agent Eval 1.0 确定性评估器。

    D1 只计算可以由 Dataset 标注 + 运行 Observation 明确推出的指标。
    不调用 LLM-as-Judge，也不猜测 Groundedness/Answerability。
    """

    EVALUATOR_VERSION = "1.0.0"

    def evaluate(
        self,
        *,
        dataset: AgentEvaluationDataset,
        dataset_reference: AgentEvaluationDatasetReference,
        observations: AgentEvaluationObservationSet,
    ) -> AgentEvaluationReport:
        observation_by_case_id = {
            observation.case_id: observation
            for observation in observations.observations
        }

        case_results = [
            self.evaluate_case(
                case=case,
                observation=observation_by_case_id[case.case_id],
            )
            for case in dataset.cases
        ]

        return AgentEvaluationReport(
            generated_at=datetime.now(timezone.utc),
            evaluator_version=self.EVALUATOR_VERSION,
            dataset=dataset_reference,
            summary=self._build_summary(case_results),
            cases=case_results,
        )

    def evaluate_case(
        self,
        *,
        case: AgentEvaluationCase,
        observation: AgentEvaluationObservation,
    ) -> AgentEvaluationCaseResult:
        if observation.case_id != case.case_id:
            raise ValueError("observation case_id does not match case")

        actual_names = [
            tool_call.tool_name
            for tool_call in observation.tool_calls
        ]
        expected_names = [
            tool_call.tool_name
            for tool_call in case.expected_tool_calls
        ]

        unauthorized_count = self._count_unauthorized_tools(
            case=case,
            actual_names=actual_names,
        )
        selection_pass = self._tool_selection_pass(
            case=case,
            actual_names=actual_names,
            unauthorized_count=unauthorized_count,
        )
        argument_accuracy = self._tool_argument_accuracy(
            expected=case.expected_tool_calls,
            actual=observation.tool_calls,
        )
        unnecessary_count = self._count_unnecessary_tool_calls(
            expected_names=expected_names,
            actual_names=actual_names,
        )
        unnecessary_rate = (
            unnecessary_count / len(actual_names)
            if actual_names
            else 0.0
        )

        answerability_match = (
            observation.answerable == case.expected_answerable
            if observation.answerable is not None
            else None
        )
        citation_correctness = self._citation_correctness(
            expected_sources=case.expected_sources,
            observed_sources=observation.observed_sources,
        )

        task_success = all(
            condition
            for condition in [
                observation.run_succeeded,
                selection_pass,
                unauthorized_count == 0,
                argument_accuracy is None or argument_accuracy == 1.0,
                answerability_match is None or answerability_match,
                observation.grounded is None or observation.grounded,
                citation_correctness is None or citation_correctness == 1.0,
            ]
        )

        return AgentEvaluationCaseResult(
            case_id=case.case_id,
            category=case.category,
            task_success=task_success,
            tool_selection_pass=selection_pass,
            tool_argument_accuracy=argument_accuracy,
            unnecessary_tool_call_rate=unnecessary_rate,
            unauthorized_tool_call_count=unauthorized_count,
            answerability_match=answerability_match,
            grounded_answer=observation.grounded,
            citation_correctness=citation_correctness,
            tool_call_count=len(actual_names),
            latency_ms=observation.latency_ms,
            input_tokens=observation.input_tokens,
            output_tokens=observation.output_tokens,
            cost=observation.cost,
        )

    @staticmethod
    def _count_unauthorized_tools(
        *,
        case: AgentEvaluationCase,
        actual_names: list[str],
    ) -> int:
        allowed = set(case.allowed_tools)
        forbidden = set(case.forbidden_tools)
        return sum(
            1
            for tool_name in actual_names
            if tool_name in forbidden or tool_name not in allowed
        )

    @staticmethod
    def _tool_selection_pass(
        *,
        case: AgentEvaluationCase,
        actual_names: list[str],
        unauthorized_count: int,
    ) -> bool:
        if unauthorized_count:
            return False

        expected_counts = Counter(
            tool_call.tool_name
            for tool_call in case.expected_tool_calls
        )
        actual_counts = Counter(actual_names)

        return all(
            actual_counts[tool_name] >= expected_count
            for tool_name, expected_count in expected_counts.items()
        ) and (
            bool(actual_names) is bool(case.expected_tool_calls)
            if not case.expected_tool_calls
            else True
        )

    @classmethod
    def _tool_argument_accuracy(
        cls,
        *,
        expected: list[AgentExpectedToolCall],
        actual: list[AgentObservedToolCall],
    ) -> float | None:
        annotated_expected = [
            tool_call
            for tool_call in expected
            if tool_call.expected_arguments
        ]
        if not annotated_expected:
            return None

        actual_by_name: dict[str, list[AgentObservedToolCall]] = {}
        for tool_call in actual:
            actual_by_name.setdefault(tool_call.tool_name, []).append(tool_call)

        name_offsets: Counter[str] = Counter()
        scores: list[float] = []

        for expected_call in expected:
            occurrence = name_offsets[expected_call.tool_name]
            name_offsets[expected_call.tool_name] += 1

            if not expected_call.expected_arguments:
                continue

            matching_actual = actual_by_name.get(expected_call.tool_name, [])
            if occurrence >= len(matching_actual):
                scores.append(0.0)
                continue

            actual_call = matching_actual[occurrence]
            scores.append(
                1.0
                if cls._is_json_subset(
                    expected_call.expected_arguments,
                    actual_call.arguments,
                )
                else 0.0
            )

        return sum(scores) / len(scores)

    @staticmethod
    def _count_unnecessary_tool_calls(
        *,
        expected_names: list[str],
        actual_names: list[str],
    ) -> int:
        remaining_expected = Counter(expected_names)
        unnecessary = 0

        for tool_name in actual_names:
            if remaining_expected[tool_name] > 0:
                remaining_expected[tool_name] -= 1
            else:
                unnecessary += 1

        return unnecessary

    @staticmethod
    def _citation_correctness(
        *,
        expected_sources: list[str],
        observed_sources: list[str],
    ) -> float | None:
        if not expected_sources:
            return None
        if not observed_sources:
            return 0.0

        expected_set = set(expected_sources)
        correct = sum(
            1
            for source in observed_sources
            if source in expected_set
        )
        return correct / len(observed_sources)

    @classmethod
    def _is_json_subset(
        cls,
        expected: Any,
        actual: Any,
    ) -> bool:
        if isinstance(expected, dict):
            if not isinstance(actual, dict):
                return False
            return all(
                key in actual and cls._is_json_subset(value, actual[key])
                for key, value in expected.items()
            )

        if isinstance(expected, list):
            if not isinstance(actual, list) or len(expected) != len(actual):
                return False
            return all(
                cls._is_json_subset(expected_item, actual_item)
                for expected_item, actual_item in zip(expected, actual)
            )

        return expected == actual

    @staticmethod
    def _build_summary(
        case_results: list[AgentEvaluationCaseResult],
    ) -> AgentEvaluationSummary:
        total_cases = len(case_results)
        total_tool_calls = sum(
            result.tool_call_count for result in case_results
        )
        unnecessary_tool_calls = sum(
            result.unnecessary_tool_call_rate * result.tool_call_count
            for result in case_results
        )

        argument_scores = [
            result.tool_argument_accuracy
            for result in case_results
            if result.tool_argument_accuracy is not None
        ]
        grounded_scores = [
            result.grounded_answer
            for result in case_results
            if result.grounded_answer is not None
        ]
        citation_scores = [
            result.citation_correctness
            for result in case_results
            if result.citation_correctness is not None
        ]

        input_tokens = [
            result.input_tokens
            for result in case_results
            if result.input_tokens is not None
        ]
        output_tokens = [
            result.output_tokens
            for result in case_results
            if result.output_tokens is not None
        ]
        costs = [
            result.cost
            for result in case_results
            if result.cost is not None
        ]

        return AgentEvaluationSummary(
            total_cases=total_cases,
            task_success_rate=sum(
                result.task_success for result in case_results
            ) / total_cases,
            tool_selection_accuracy=sum(
                result.tool_selection_pass for result in case_results
            ) / total_cases,
            tool_argument_accuracy=(
                sum(argument_scores) / len(argument_scores)
                if argument_scores
                else None
            ),
            unnecessary_tool_call_rate=(
                unnecessary_tool_calls / total_tool_calls
                if total_tool_calls
                else 0.0
            ),
            unauthorized_tool_call_count=sum(
                result.unauthorized_tool_call_count
                for result in case_results
            ),
            grounded_answer_rate=(
                sum(grounded_scores) / len(grounded_scores)
                if grounded_scores
                else None
            ),
            citation_correctness=(
                sum(citation_scores) / len(citation_scores)
                if citation_scores
                else None
            ),
            average_tool_calls=total_tool_calls / total_cases,
            average_latency_ms=sum(
                result.latency_ms for result in case_results
            ) / total_cases,
            total_input_tokens=(sum(input_tokens) if input_tokens else None),
            total_output_tokens=(
                sum(output_tokens) if output_tokens else None
            ),
            total_cost=(sum(costs) if costs else None),
        )
