"""比较 Native Baseline 与 LangChain Candidate 的 Agent Eval 结果。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.schemas.agent_evaluation import (
    AgentEvaluationObservationSet,
    AgentEvaluationReport,
)
from app.services.evaluation.agent_runtime_comparison_service import (
    AgentRuntimeComparisonService,
)


DEFAULT_BASELINE_REPORT = Path(
    "evaluation/reports/agent_live_evaluation_v1.json"
)
DEFAULT_BASELINE_OBSERVATIONS = Path(
    "evaluation/reports/agent_live_observations_v1.json"
)
DEFAULT_CANDIDATE_REPORT = Path(
    "evaluation/reports/langchain_live_evaluation_v1.json"
)
DEFAULT_CANDIDATE_OBSERVATIONS = Path(
    "evaluation/reports/langchain_live_observations_v1.json"
)
DEFAULT_OUTPUT = Path(
    "evaluation/reports/agent_runtime_comparison_v1.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Native Agent baseline and LangChain candidate using the "
            "same Agent Eval dataset/evaluator contract."
        )
    )
    parser.add_argument(
        "--baseline-report",
        type=Path,
        default=DEFAULT_BASELINE_REPORT,
    )
    parser.add_argument(
        "--baseline-observations",
        type=Path,
        default=DEFAULT_BASELINE_OBSERVATIONS,
    )
    parser.add_argument(
        "--candidate-report",
        type=Path,
        default=DEFAULT_CANDIDATE_REPORT,
    )
    parser.add_argument(
        "--candidate-observations",
        type=Path,
        default=DEFAULT_CANDIDATE_OBSERVATIONS,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Return exit code 2 when the comparison decision is fail.",
    )
    parser.add_argument(
        "--fail-on-inconclusive",
        action="store_true",
        help="Return exit code 3 when Groundedness coverage is incomplete.",
    )
    return parser.parse_args()


def _load_model(path: Path, model_type):
    return model_type.model_validate_json(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()

    baseline_report = _load_model(
        args.baseline_report,
        AgentEvaluationReport,
    )
    baseline_observations = _load_model(
        args.baseline_observations,
        AgentEvaluationObservationSet,
    )
    candidate_report = _load_model(
        args.candidate_report,
        AgentEvaluationReport,
    )
    candidate_observations = _load_model(
        args.candidate_observations,
        AgentEvaluationObservationSet,
    )

    comparison = AgentRuntimeComparisonService().compare(
        baseline_report=baseline_report,
        baseline_observations=baseline_observations,
        candidate_report=candidate_report,
        candidate_observations=candidate_observations,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        comparison.model_dump_json(indent=2),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "decision": comparison.summary.decision,
                "dataset_id": comparison.dataset.dataset_id,
                "dataset_version": comparison.dataset.dataset_version,
                "evaluator_version": comparison.evaluator_version,
                "baseline_runner_version": comparison.baseline_runner_version,
                "candidate_runner_version": comparison.candidate_runner_version,
                "deterministic_gate_passed": (
                    comparison.summary.deterministic_gate_passed
                ),
                "groundedness_gate_status": (
                    comparison.summary.groundedness_gate_status
                ),
                "failed_metrics": comparison.summary.failed_metrics,
                "inconclusive_metrics": (
                    comparison.summary.inconclusive_metrics
                ),
                "regression_case_ids": (
                    comparison.summary.regression_case_ids
                ),
                "improvement_case_ids": (
                    comparison.summary.improvement_case_ids
                ),
                "task_success_rate_delta": (
                    comparison.summary.task_success_rate_delta
                ),
                "average_tool_calls_delta": (
                    comparison.summary.average_tool_calls_delta
                ),
                "average_latency_ms_delta": (
                    comparison.summary.average_latency_ms_delta
                ),
                "average_latency_ratio": (
                    comparison.summary.average_latency_ratio
                ),
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    if args.fail_on_regression and comparison.summary.decision == "fail":
        return 2
    if (
        args.fail_on_inconclusive
        and comparison.summary.decision == "inconclusive"
    ):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
