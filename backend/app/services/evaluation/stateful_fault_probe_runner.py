from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from app.schemas.stateful_fault_verification import (
    StatefulFaultNodeObservation,
    StatefulFaultObservationSet,
    StatefulFaultSuiteDefinition,
)


class StatefulFaultProbeExecutionError(RuntimeError):
    """pytest probe 本身无法启动或结果文件不可解析。"""


class StatefulFaultPytestProbeRunner:
    """一次 pytest 进程执行全部 A12 Stateful probe，并读取 JUnit 观察值。"""

    def run(
        self,
        *,
        suite: StatefulFaultSuiteDefinition,
        project_root: Path,
        extra_env: dict[str, str] | None = None,
        timeout_seconds: float = 180.0,
    ) -> StatefulFaultObservationSet:
        nodeids = tuple(
            dict.fromkeys(
                nodeid
                for case in suite.cases
                for nodeid in case.required_nodeids
            )
        )

        with tempfile.TemporaryDirectory(prefix="stateful-fault-") as temp_dir:
            junit_path = Path(temp_dir) / "pytest-junit.xml"
            command = [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                *nodeids,
                f"--junitxml={junit_path}",
            ]
            env = {**os.environ, **(extra_env or {})}
            started = time.perf_counter()
            try:
                completed = subprocess.run(
                    command,
                    cwd=project_root,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise StatefulFaultProbeExecutionError(
                    "stateful fault pytest probe could not complete"
                ) from exc
            elapsed_ms = (time.perf_counter() - started) * 1000

            if not junit_path.is_file():
                detail = (completed.stderr or completed.stdout).strip()
                raise StatefulFaultProbeExecutionError(
                    "stateful fault pytest probe did not produce JUnit XML: "
                    + detail[-1500:]
                )

            observations = self._parse_junit(
                junit_path,
                expected_nodeids=nodeids,
            )

        if observations and all(item.duration_ms == 0 for item in observations):
            fallback = elapsed_ms / len(observations)
            observations = tuple(
                item.model_copy(update={"duration_ms": fallback})
                for item in observations
            )

        return StatefulFaultObservationSet(
            suite_id=suite.suite_id,
            suite_version=suite.suite_version,
            generated_at=datetime.now(timezone.utc),
            pytest_exit_code=completed.returncode,
            observations=observations,
        )

    def _parse_junit(
        self,
        path: Path,
        *,
        expected_nodeids: tuple[str, ...],
    ) -> tuple[StatefulFaultNodeObservation, ...]:
        try:
            root = ET.parse(path).getroot()
        except (ET.ParseError, OSError) as exc:
            raise StatefulFaultProbeExecutionError(
                "invalid stateful fault JUnit XML"
            ) from exc

        by_function: dict[str, str] = {
            nodeid.rsplit("::", 1)[1].split("[", 1)[0]: nodeid
            for nodeid in expected_nodeids
        }
        results: dict[str, StatefulFaultNodeObservation] = {}

        for testcase in root.iter("testcase"):
            name = (testcase.get("name") or "").split("[", 1)[0]
            nodeid = by_function.get(name)
            if nodeid is None:
                continue

            duration_ms = max(
                0.0,
                float(testcase.get("time") or 0.0) * 1000,
            )
            failure = testcase.find("failure")
            error = testcase.find("error")
            skipped = testcase.find("skipped")

            if failure is not None or error is not None:
                detail_node = failure if failure is not None else error
                detail = ""
                if detail_node is not None:
                    detail = (
                        detail_node.get("message")
                        or detail_node.text
                        or "pytest case failed"
                    ).strip()
                status = "fail"
                failure_message = detail[-2000:] or "pytest case failed"
            elif skipped is not None:
                status = "skipped"
                failure_message = None
            else:
                status = "pass"
                failure_message = None

            results[nodeid] = StatefulFaultNodeObservation(
                nodeid=nodeid,
                status=status,
                duration_ms=duration_ms,
                failure_message=failure_message,
            )

        for nodeid in expected_nodeids:
            results.setdefault(
                nodeid,
                StatefulFaultNodeObservation(
                    nodeid=nodeid,
                    status="fail",
                    duration_ms=0.0,
                    failure_message=(
                        "pytest did not execute this required nodeid; "
                        "check collection/import errors"
                    ),
                ),
            )

        return tuple(results[nodeid] for nodeid in expected_nodeids)
