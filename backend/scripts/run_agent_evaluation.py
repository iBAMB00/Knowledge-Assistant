import argparse
import json
from pathlib import Path

from app.services.evaluation.agent_case_loader import AgentEvaluationCaseLoader
from app.services.evaluation.agent_evaluator import AgentEvaluator


DEFAULT_CASES_PATH = Path("evaluation/agent_cases.json")
DEFAULT_REPORT_PATH = Path("evaluation/reports/agent_evaluation_v1.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the versioned Agent Eval dataset and, when "
            "observations are supplied, calculate deterministic Agent "
            "Eval 1.0 metrics."
        )
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_CASES_PATH,
    )
    parser.add_argument(
        "--observations",
        type=Path,
        default=None,
        help=(
            "JSON observations produced by a real/fake Agent run. "
            "Omit together with --validate-only to validate the dataset only."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REPORT_PATH,
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
    )

    args = parser.parse_args()
    if not args.validate_only and args.observations is None:
        parser.error("--observations is required unless --validate-only is set")
    return args


def main() -> int:
    args = parse_args()
    loader = AgentEvaluationCaseLoader()
    dataset = loader.load_dataset(args.cases)
    dataset_reference = loader.build_reference(dataset, args.cases)

    if args.validate_only:
        print(
            json.dumps(
                {
                    "dataset_id": dataset.dataset_id,
                    "dataset_version": dataset.dataset_version,
                    "total_cases": len(dataset.cases),
                    "source_sha256": dataset_reference.source_sha256,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    observations = loader.load_observations(args.observations)
    loader.validate_observation_coverage(dataset, observations)

    report = AgentEvaluator().evaluate(
        dataset=dataset,
        dataset_reference=dataset_reference,
        observations=observations,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        report.model_dump_json(indent=2),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "output": str(args.output),
                "task_success_rate": report.summary.task_success_rate,
                "tool_selection_accuracy": (
                    report.summary.tool_selection_accuracy
                ),
                "unauthorized_tool_call_count": (
                    report.summary.unauthorized_tool_call_count
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
