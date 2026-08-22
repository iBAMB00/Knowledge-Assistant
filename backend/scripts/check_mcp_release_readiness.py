"""组合 MCP Live Evidence 与 Agent Runtime Comparison，执行 v2.2 Release Gate。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.schemas.agent_runtime_comparison import AgentRuntimeComparisonReport
from app.schemas.mcp_release import MCPLiveVerificationReport
from app.services.evaluation.mcp_release_readiness_service import (
    MCPReleaseReadinessService,
)


DEFAULT_LIVE = Path("evaluation/reports/mcp_live_verification_v1.json")
DEFAULT_COMPARISON = Path("evaluation/reports/agent_runtime_comparison_v2_2.json")
DEFAULT_OUTPUT = Path("evaluation/reports/mcp_release_gate_v1.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check v2.2 MCP release readiness.")
    parser.add_argument("--live", type=Path, default=DEFAULT_LIVE)
    parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fail-on-not-ready", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    live = MCPLiveVerificationReport.model_validate_json(
        args.live.read_text(encoding="utf-8")
    )
    comparison = AgentRuntimeComparisonReport.model_validate_json(
        args.comparison.read_text(encoding="utf-8")
    )
    result = MCPReleaseReadinessService().evaluate(
        live_report=live,
        comparison=comparison,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result.model_dump_json(indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "decision": result.decision,
                "release_ready": result.release_ready,
                "mcp_tool_name": result.mcp_tool_name,
                "toolset_version": result.toolset_version,
                "dataset_id": result.dataset_id,
                "dataset_version": result.dataset_version,
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
