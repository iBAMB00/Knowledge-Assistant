"""Run the deterministic v2.4 Context / Memory release gate.

This verifier is evaluation-only. It executes one focused pytest node per frozen
v2.4 acceptance criterion and writes a structured JSON report without changing
production runtime behavior.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

SUITE_VERSION = "1.0.0"


@dataclass(frozen=True)
class GateCase:
    case_id: str
    title: str
    nodeid: str


@dataclass(frozen=True)
class GateResult:
    case_id: str
    title: str
    decision: str
    nodeid: str
    exit_code: int | None
    summary: str


CASES = (
    GateCase(
        "short_conversation_no_regression",
        "短对话不回退",
        "tests/test_context_memory_release_gate.py::test_release_short_conversation_has_no_context_regression",
    ),
    GateCase(
        "long_conversation_continuity",
        "长对话连续性",
        "tests/test_context_memory_release_gate.py::test_release_long_conversation_preserves_summary_memory_and_recent_history",
    ),
    GateCase(
        "history_truncation",
        "History Truncation",
        "tests/test_context_memory_release_gate.py::test_release_history_truncation_keeps_recent_messages",
    ),
    GateCase(
        "summary_continuity",
        "Summary Continuity",
        "tests/test_context_memory_release_gate.py::test_release_summary_continuity_is_incremental_and_keeps_recent_raw_messages",
    ),
    GateCase(
        "memory_save_recall",
        "Memory Save / Recall",
        "tests/test_context_memory_release_gate.py::test_release_memory_save_recall_baseline_returns_relevant_active_memory",
    ),
    GateCase(
        "wrong_memory_not_recalled",
        "错误 Memory 不误导",
        "tests/test_context_memory_release_gate.py::test_release_irrelevant_memory_is_not_recalled_for_unrelated_query",
    ),
    GateCase(
        "knowledge_over_memory_trust",
        "Knowledge > Memory trust",
        "tests/test_context_memory_release_gate.py::test_release_enterprise_knowledge_outranks_memory_under_budget_and_prompt_trust",
    ),
    GateCase(
        "user_conversation_isolation",
        "User / Conversation isolation",
        "tests/test_context_memory_release_gate.py::test_release_memory_provider_keeps_user_and_conversation_scope_explicit",
    ),
    GateCase(
        "runtime_context_parity",
        "Native / LangChain / LangGraph Context Parity",
        "tests/test_context_memory_release_gate.py::test_release_runtime_parity_uses_framework_neutral_context_contract",
    ),
    GateCase(
        "context_growth_bounded",
        "Token / Context Growth",
        "tests/test_context_memory_release_gate.py::test_release_context_growth_stays_bounded_when_history_grows",
    ),
)


def parse_args() -> argparse.Namespace:
    default_backend = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run Knowledge Assistant v2.4 Context / Memory verification."
    )
    parser.add_argument("--backend-dir", type=Path, default=default_backend)
    parser.add_argument(
        "--output",
        type=Path,
        default=default_backend
        / "evaluation"
        / "reports"
        / "context_memory_verification_v1.json",
    )
    parser.add_argument("--timeout-seconds", type=float, default=90.0)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--fail-on-not-ready", action="store_true")
    return parser.parse_args()


def _summarize(value: str, *, max_chars: int = 3000) -> str:
    lines = [line.rstrip() for line in value.splitlines() if line.strip()]
    return "\n".join(lines[-20:])[-max_chars:]


def _run_case(
    case: GateCase,
    *,
    backend_dir: Path,
    timeout_seconds: float,
) -> GateResult:
    command = [sys.executable, "-m", "pytest", "-q", case.nodeid]
    try:
        completed = subprocess.run(
            command,
            cwd=backend_dir,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        combined = "\n".join(
            part
            for part in (completed.stdout, completed.stderr)
            if part and part.strip()
        )
        return GateResult(
            case_id=case.case_id,
            title=case.title,
            decision="pass" if completed.returncode == 0 else "fail",
            nodeid=case.nodeid,
            exit_code=completed.returncode,
            summary=_summarize(combined),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return GateResult(
            case_id=case.case_id,
            title=case.title,
            decision="fail",
            nodeid=case.nodeid,
            exit_code=None,
            summary=f"{type(exc).__name__}: {exc}",
        )


def main() -> int:
    args = parse_args()
    backend_dir = args.backend_dir.resolve()

    if args.plan:
        print("v2.4 Context / Memory Verification Plan")
        for case in CASES:
            print(f"- {case.case_id}: {case.nodeid}")
        return 0

    results = [
        _run_case(
            case,
            backend_dir=backend_dir,
            timeout_seconds=args.timeout_seconds,
        )
        for case in CASES
    ]
    failed = [result.case_id for result in results if result.decision != "pass"]
    decision = "pass" if not failed else "fail"

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "suite_version": SUITE_VERSION,
        "release_version": "2.4.0",
        "decision": decision,
        "release_ready": decision == "pass",
        "passed_cases": len(results) - len(failed),
        "total_cases": len(results),
        "failed_cases": failed,
        "cases": [asdict(result) for result in results],
        "notes": [
            "The gate is deterministic and does not call a live LLM.",
            "Live model quality and frontend behavior remain release smoke checks.",
            "Knowledge > Memory trust is a release-blocking invariant.",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "decision": decision,
                "release_ready": report["release_ready"],
                "passed_cases": report["passed_cases"],
                "total_cases": report["total_cases"],
                "failed_cases": failed,
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    if args.fail_on_not_ready and decision != "pass":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
