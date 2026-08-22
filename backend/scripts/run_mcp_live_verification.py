"""运行真实 MCP stdio 发布探针并保存 v2.2 Release Evidence。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.services.evaluation.mcp_live_verification_service import (
    MCPLiveVerificationService,
)


DEFAULT_SERVER = Path("scripts/mcp_release_probe_server.py")
DEFAULT_OUTPUT = Path("evaluation/reports/mcp_live_verification_v1.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the v2.2 real MCP stdio probe.")
    parser.add_argument("--server", type=Path, default=DEFAULT_SERVER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fail-on-error", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = MCPLiveVerificationService().run(server_script=args.server)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "decision": report.decision,
                "sdk_version": report.sdk_version,
                "transport": report.transport,
                "server_id": report.server_id,
                "exposed_tool_name": report.exposed_tool_name,
                "tool_contract_version": report.tool_contract_version,
                "mcp_toolset_version": report.mcp_toolset_version,
                "discovery_succeeded": report.discovery_succeeded,
                "dispatch_succeeded": report.dispatch_succeeded,
                "structured_output_succeeded": report.structured_output_succeeded,
                "failure_reason": report.failure_reason,
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 2 if args.fail_on_error and report.decision != "pass" else 0


if __name__ == "__main__":
    raise SystemExit(main())
