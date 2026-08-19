"""检查当前 v2.1 Framework Candidate 是否拥有新鲜的回归门禁证据。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.core.config import get_settings
from app.schemas.agent_runtime_comparison import AgentRuntimeComparisonReport
from app.services.agent_runtime_diagnostics_service import (
    AgentRuntimeDiagnosticsService,
)


DEFAULT_COMPARISON = Path(
    "evaluation/reports/agent_runtime_comparison_v1.json"
)
DEFAULT_OUTPUT = Path(
    "evaluation/reports/agent_framework_release_gate_v1.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate that the current LangChain candidate code has a "
            "matching PASS runtime-comparison report."
        )
    )
    parser.add_argument(
        "--comparison",
        type=Path,
        default=DEFAULT_COMPARISON,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--fail-on-not-ready",
        action="store_true",
        help=(
            "Return exit code 2 for fail and 3 for inconclusive release "
            "decisions."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    comparison = AgentRuntimeComparisonReport.model_validate_json(
        args.comparison.read_text(encoding="utf-8")
    )
    settings = get_settings()
    service = AgentRuntimeDiagnosticsService(
        langchain_candidate_enabled=(
            settings.agent_langchain_candidate_enabled
        )
    )
    result = service.evaluate_framework_release(comparison)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        result.model_dump_json(indent=2),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "decision": result.decision,
                "release_ready": result.release_ready,
                "comparison_decision": result.comparison_decision,
                "dataset_id": result.dataset_id,
                "dataset_version": result.dataset_version,
                "evaluator_version": result.evaluator_version,
                "expected_evaluator_version": (
                    result.expected_evaluator_version
                ),
                "baseline_runner_version": result.baseline_runner_version,
                "expected_baseline_runner_version": (
                    result.expected_baseline_runner_version
                ),
                "candidate_runner_version": result.candidate_runner_version,
                "expected_candidate_runner_version": (
                    result.expected_candidate_runner_version
                ),
                "candidate_feature_gate_enabled": (
                    result.candidate_feature_gate_enabled
                ),
                "reasons": result.reasons,
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    if args.fail_on_not_ready:
        if result.decision == "fail":
            return 2
        if result.decision == "inconclusive":
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
