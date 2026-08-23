"""执行 v2.3 Stateful / LangGraph Fault & Recovery 验证套件。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agent.checkpoint import CHECKPOINT_SCHEMA_VERSION
from app.agent.frameworks.langgraph.runner import LangGraphStatefulRunner
from app.agent.state import AGENT_STATE_SCHEMA_VERSION
from app.services.evaluation.stateful_fault_case_loader import (
    StatefulFaultCaseLoader,
)
from app.services.evaluation.stateful_fault_evaluator import (
    StatefulFaultEvaluator,
)
from app.services.evaluation.stateful_fault_probe_runner import (
    StatefulFaultPytestProbeRunner,
)


DEFAULT_SUITE = Path("evaluation/stateful_fault_cases.json")
DEFAULT_OBSERVATIONS = Path(
    "evaluation/reports/stateful_fault_observations_v1.json"
)
DEFAULT_REPORT = Path(
    "evaluation/reports/stateful_fault_verification_v1.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the versioned v2.3 Stateful fault/recovery pytest probes "
            "and persist auditable observations/report evidence."
        )
    )
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument(
        "--observations-output",
        type=Path,
        default=DEFAULT_OBSERVATIONS,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate suite schema and referenced pytest nodeids only.",
    )
    parser.add_argument(
        "--fail-on-not-ready",
        action="store_true",
        help=(
            "Return exit code 2 for fail and 3 for inconclusive report decisions."
        ),
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=180.0,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = PROJECT_ROOT
    loader = StatefulFaultCaseLoader()
    suite = loader.load(args.suite)
    loader.validate_nodeids_exist(suite, project_root=project_root)

    if args.validate_only:
        print(
            json.dumps(
                {
                    "suite_id": suite.suite_id,
                    "suite_version": suite.suite_version,
                    "total_cases": len(suite.cases),
                    "total_pytest_nodeids": len(
                        {
                            nodeid
                            for case in suite.cases
                            for nodeid in case.required_nodeids
                        }
                    ),
                    "decision": "valid",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    observations = StatefulFaultPytestProbeRunner().run(
        suite=suite,
        project_root=project_root,
        timeout_seconds=args.timeout_seconds,
    )
    report = StatefulFaultEvaluator().evaluate(
        suite=suite,
        observations=observations,
        runner_version=LangGraphStatefulRunner.RUNNER_VERSION,
        graph_version=LangGraphStatefulRunner.GRAPH_VERSION,
        checkpoint_schema_version=CHECKPOINT_SCHEMA_VERSION,
        state_schema_version=AGENT_STATE_SCHEMA_VERSION,
    )

    args.observations_output.parent.mkdir(parents=True, exist_ok=True)
    args.observations_output.write_text(
        observations.model_dump_json(indent=2),
        encoding="utf-8",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        report.model_dump_json(indent=2),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "decision": report.summary.decision,
                "suite_id": report.suite_id,
                "suite_version": report.suite_version,
                "runner_version": report.runner_version,
                "graph_version": report.graph_version,
                "checkpoint_schema_version": (
                    report.checkpoint_schema_version
                ),
                "state_schema_version": report.state_schema_version,
                "total_cases": report.summary.total_cases,
                "passed_cases": report.summary.passed_cases,
                "failed_cases": report.summary.failed_cases,
                "skipped_cases": report.summary.skipped_cases,
                "pass_rate": report.summary.pass_rate,
                "failed_case_ids": report.summary.failed_case_ids,
                "skipped_case_ids": report.summary.skipped_case_ids,
                "pytest_exit_code": report.pytest_exit_code,
                "observations_output": str(args.observations_output),
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    if args.fail_on_not_ready:
        if report.summary.decision == "fail":
            return 2
        if report.summary.decision == "inconclusive":
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
