from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.schemas.stateful_fault_verification import (
    StatefulFaultNodeObservation,
    StatefulFaultObservationSet,
)
from app.services.evaluation.stateful_fault_case_loader import (
    StatefulFaultCaseLoader,
    StatefulFaultSuiteValidationError,
)
from app.services.evaluation.stateful_fault_evaluator import (
    StatefulFaultEvaluator,
    StatefulFaultObservationCoverageError,
)


SUITE_PATH = Path("evaluation/stateful_fault_cases.json")


def _observations(status_by_nodeid: dict[str, str] | None = None):
    suite = StatefulFaultCaseLoader().load(SUITE_PATH)
    status_by_nodeid = status_by_nodeid or {}
    nodeids = tuple(
        dict.fromkeys(
            nodeid
            for case in suite.cases
            for nodeid in case.required_nodeids
        )
    )
    return suite, StatefulFaultObservationSet(
        suite_id=suite.suite_id,
        suite_version=suite.suite_version,
        generated_at=datetime.now(timezone.utc),
        pytest_exit_code=0,
        observations=tuple(
            StatefulFaultNodeObservation(
                nodeid=nodeid,
                status=status_by_nodeid.get(nodeid, "pass"),
                duration_ms=1.0,
                failure_message=(
                    "forced failure"
                    if status_by_nodeid.get(nodeid) == "fail"
                    else None
                ),
            )
            for nodeid in nodeids
        ),
    )


def test_stateful_fault_suite_is_versioned_and_references_existing_tests() -> None:
    loader = StatefulFaultCaseLoader()
    suite = loader.load(SUITE_PATH)

    loader.validate_nodeids_exist(suite, project_root=Path.cwd())

    assert suite.suite_id == "knowledge-assistant-stateful-fault-verification"
    assert suite.suite_version == "1.0.0"
    assert len(suite.cases) == 11
    assert {case.category for case in suite.cases} >= {
        "checkpoint",
        "recovery",
        "hitl",
        "cancellation",
        "isolation",
        "compatibility",
        "runtime_parity",
        "production_api",
    }


def test_stateful_fault_evaluator_passes_only_when_every_case_passes() -> None:
    suite, observations = _observations()

    report = StatefulFaultEvaluator().evaluate(
        suite=suite,
        observations=observations,
        runner_version="runner-test",
        graph_version="graph-test",
        checkpoint_schema_version="1.1",
        state_schema_version="1.0",
    )

    assert report.summary.decision == "pass"
    assert report.summary.total_cases == 11
    assert report.summary.passed_cases == 11
    assert report.summary.failed_cases == 0
    assert report.summary.skipped_cases == 0
    assert report.summary.pass_rate == 1.0


def test_stateful_fault_evaluator_maps_failed_node_to_failed_case() -> None:
    suite = StatefulFaultCaseLoader().load(SUITE_PATH)
    target = suite.cases[2].required_nodeids[0]
    suite, observations = _observations({target: "fail"})

    report = StatefulFaultEvaluator().evaluate(
        suite=suite,
        observations=observations,
        runner_version="runner-test",
        graph_version="graph-test",
        checkpoint_schema_version="1.1",
        state_schema_version="1.0",
    )

    assert report.summary.decision == "fail"
    assert suite.cases[2].case_id in report.summary.failed_case_ids
    failed_case = next(
        item for item in report.cases if item.case_id == suite.cases[2].case_id
    )
    assert failed_case.status == "fail"
    assert failed_case.failure_reasons == ("forced failure",)


def test_stateful_fault_evaluator_marks_skipped_case_inconclusive() -> None:
    suite = StatefulFaultCaseLoader().load(SUITE_PATH)
    target = suite.cases[-1].required_nodeids[0]
    suite, observations = _observations({target: "skipped"})

    report = StatefulFaultEvaluator().evaluate(
        suite=suite,
        observations=observations,
        runner_version="runner-test",
        graph_version="graph-test",
        checkpoint_schema_version="1.1",
        state_schema_version="1.0",
    )

    assert report.summary.decision == "inconclusive"
    assert suite.cases[-1].case_id in report.summary.skipped_case_ids


def test_stateful_fault_evaluator_rejects_incomplete_observations() -> None:
    suite, observations = _observations()
    incomplete = observations.model_copy(
        update={"observations": observations.observations[:-1]}
    )

    with pytest.raises(
        StatefulFaultObservationCoverageError,
        match="missing stateful fault observations",
    ):
        StatefulFaultEvaluator().evaluate(
            suite=suite,
            observations=incomplete,
            runner_version="runner-test",
            graph_version="graph-test",
            checkpoint_schema_version="1.1",
            state_schema_version="1.0",
        )


def test_stateful_fault_loader_rejects_missing_nodeid(tmp_path: Path) -> None:
    loader = StatefulFaultCaseLoader()
    suite = loader.load(SUITE_PATH)
    broken = suite.model_copy(
        update={
            "cases": (
                suite.cases[0].model_copy(
                    update={
                        "required_nodeids": (
                            "tests/test_missing.py::test_missing",
                        )
                    }
                ),
                *suite.cases[1:],
            )
        }
    )

    with pytest.raises(
        StatefulFaultSuiteValidationError,
        match="missing pytest nodeids",
    ):
        loader.validate_nodeids_exist(broken, project_root=Path.cwd())
