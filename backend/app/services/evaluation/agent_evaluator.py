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

    EVALUATOR_VERSION = "1.4.0"

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

        policy_violation_count = self._count_tool_policy_violations(
            case=case,
            actual_names=actual_names,
        )
        selection_pass = self._tool_selection_pass(
            case=case,
            actual_names=actual_names,
            policy_violation_count=policy_violation_count,
        )
        execution_pass = self._tool_execution_pass(
            expected=case.expected_tool_calls,
            actual=observation.tool_calls,
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
            retrieved_sources=observation.retrieved_sources,
            observed_sources=observation.observed_sources,
            require_citation=case.require_citation,
        )
        retrieved_evidence_pass = (
            bool(observation.retrieved_sources)
            if case.require_retrieved_evidence
            else None
        )
        citation_requirement_pass = (
            bool(observation.observed_sources)
            and citation_correctness == 1.0
            if case.require_citation
            else None
        )

        task_success = all(
            condition
            for condition in [
                observation.run_succeeded,
                selection_pass,
                execution_pass is None or execution_pass,
                policy_violation_count == 0,
                unnecessary_count == 0,
                argument_accuracy is None or argument_accuracy == 1.0,
                answerability_match is None or answerability_match,
                (
                    not case.evaluate_groundedness
                    or observation.grounded is True
                ),
                (
                    retrieved_evidence_pass is None
                    or retrieved_evidence_pass
                ),
                (
                    citation_requirement_pass is None
                    or citation_requirement_pass
                ),
                citation_correctness is None or citation_correctness == 1.0,
            ]
        )

        return AgentEvaluationCaseResult(
            case_id=case.case_id,
            category=case.category,
            task_success=task_success,
            tool_selection_pass=selection_pass,
            tool_execution_pass=execution_pass,
            tool_argument_accuracy=argument_accuracy,
            unnecessary_tool_call_rate=unnecessary_rate,
            tool_policy_violation_count=policy_violation_count,
            answerability_match=answerability_match,
            groundedness_applicable=case.evaluate_groundedness,
            grounded_answer=observation.grounded,
            groundedness_score=observation.grounded_score,
            groundedness_judge_error_type=(
                observation.grounded_judge_error_type
            ),
            retrieved_evidence_pass=retrieved_evidence_pass,
            citation_requirement_pass=citation_requirement_pass,
            citation_correctness=citation_correctness,
            tool_call_count=len(actual_names),
            latency_ms=observation.latency_ms,
            input_tokens=observation.input_tokens,
            output_tokens=observation.output_tokens,
            cost=observation.cost,
        )

    @staticmethod
    def _count_tool_policy_violations(
        *,
        case: AgentEvaluationCase,
        actual_names: list[str],
    ) -> int:
        """统计模型违反当前 Eval Case Tool Policy 的决策次数。"""
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
        policy_violation_count: int,
    ) -> bool:
        if policy_violation_count:
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

    @staticmethod
    def _tool_execution_pass(
        *,
        expected: list[AgentExpectedToolCall],
        actual: list[AgentObservedToolCall],
    ) -> bool | None:
        """
        校验期望 Tool 是否得到期望执行结果。

        expected_error_code=None 表示 Tool 应成功；显式错误码表示该 Case
        期望 Runtime 以对应安全错误收口，例如 resource_not_found。
        NO_TOOL / INJECTION 等无期望 Tool 的 Case 返回 None，不参与聚合。
        """

        if not expected:
            return None

        actual_by_name: dict[str, list[AgentObservedToolCall]] = {}
        for tool_call in actual:
            actual_by_name.setdefault(tool_call.tool_name, []).append(tool_call)

        name_offsets: Counter[str] = Counter()
        for expected_call in expected:
            occurrence = name_offsets[expected_call.tool_name]
            name_offsets[expected_call.tool_name] += 1

            matching_actual = actual_by_name.get(expected_call.tool_name, [])
            if occurrence >= len(matching_actual):
                return False

            actual_call = matching_actual[occurrence]
            if actual_call.error_code != expected_call.expected_error_code:
                return False

        return True

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
        retrieved_sources: list[str],
        observed_sources: list[str],
        require_citation: bool,
    ) -> float | None:
        """评估“已经出现的 Citation 是否真实”，并与 Citation Requirement 解耦。

        Citation Requirement 回答“该引用时有没有引用”；Citation Correctness
        只回答“已经引用的 source_ref 是否属于 Ground Truth / 本次检索证据”。
        因此 NO_ANSWER 等不要求引用的 Case 即使 Retrieval 返回了低相关证据，
        只要最终回答没有引用，就保持不可评估(None)，而不是误记为 0。
        """

        reference_sources = expected_sources or retrieved_sources

        if not observed_sources:
            if require_citation and reference_sources:
                return 0.0
            return None

        if not reference_sources:
            return 0.0

        reference_set = set(reference_sources)
        correct = sum(
            1
            for source in observed_sources
            if source in reference_set
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

        execution_scores = [
            result.tool_execution_pass
            for result in case_results
            if result.tool_execution_pass is not None
        ]
        argument_scores = [
            result.tool_argument_accuracy
            for result in case_results
            if result.tool_argument_accuracy is not None
        ]
        grounded_applicable = [
            result
            for result in case_results
            if result.groundedness_applicable
        ]
        grounded_scores = [
            result.grounded_answer
            for result in grounded_applicable
            if result.grounded_answer is not None
        ]
        evidence_requirement_scores = [
            result.retrieved_evidence_pass
            for result in case_results
            if result.retrieved_evidence_pass is not None
        ]
        citation_requirement_scores = [
            result.citation_requirement_pass
            for result in case_results
            if result.citation_requirement_pass is not None
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
            tool_execution_accuracy=(
                sum(execution_scores) / len(execution_scores)
                if execution_scores
                else None
            ),
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
            tool_policy_violation_count=sum(
                result.tool_policy_violation_count
                for result in case_results
            ),
            grounded_answer_rate=(
                sum(grounded_scores) / len(grounded_scores)
                if grounded_scores
                else None
            ),
            groundedness_coverage=(
                len(grounded_scores) / len(grounded_applicable)
                if grounded_applicable
                else None
            ),
            required_evidence_success_rate=(
                sum(evidence_requirement_scores)
                / len(evidence_requirement_scores)
                if evidence_requirement_scores
                else None
            ),
            required_citation_success_rate=(
                sum(citation_requirement_scores)
                / len(citation_requirement_scores)
                if citation_requirement_scores
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
