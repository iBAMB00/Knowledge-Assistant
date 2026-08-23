from __future__ import annotations

from datetime import datetime, timezone

from app.schemas.stateful_fault_verification import (
    StatefulFaultCaseResult,
    StatefulFaultObservationSet,
    StatefulFaultSuiteDefinition,
    StatefulFaultVerificationReport,
    StatefulFaultVerificationSummary,
)


class StatefulFaultObservationCoverageError(ValueError):
    """Probe 观察值没有完整覆盖版本化验证套件。"""


class StatefulFaultEvaluator:
    """将 pytest probe 观察值聚合为 v2.3 Stateful Fault 验证报告。"""

    EVALUATOR_VERSION = "1.0.0"

    def evaluate(
        self,
        *,
        suite: StatefulFaultSuiteDefinition,
        observations: StatefulFaultObservationSet,
        runner_version: str,
        graph_version: str,
        checkpoint_schema_version: str,
        state_schema_version: str,
    ) -> StatefulFaultVerificationReport:
        if observations.suite_id != suite.suite_id:
            raise StatefulFaultObservationCoverageError(
                "observation suite_id does not match suite"
            )
        if observations.suite_version != suite.suite_version:
            raise StatefulFaultObservationCoverageError(
                "observation suite_version does not match suite"
            )

        by_nodeid = {item.nodeid: item for item in observations.observations}
        required_nodeids = {
            nodeid
            for case in suite.cases
            for nodeid in case.required_nodeids
        }
        missing = sorted(required_nodeids - set(by_nodeid))
        if missing:
            raise StatefulFaultObservationCoverageError(
                "missing stateful fault observations: " + ", ".join(missing)
            )

        case_results: list[StatefulFaultCaseResult] = []
        for case in suite.cases:
            node_results = [by_nodeid[nodeid] for nodeid in case.required_nodeids]
            statuses = {item.status for item in node_results}
            if "fail" in statuses:
                status = "fail"
            elif "skipped" in statuses:
                status = "skipped"
            else:
                status = "pass"

            failure_reasons = tuple(
                item.failure_message or f"{item.nodeid} failed"
                for item in node_results
                if item.status == "fail"
            )
            case_results.append(
                StatefulFaultCaseResult(
                    case_id=case.case_id,
                    category=case.category,
                    description=case.description,
                    status=status,
                    duration_ms=sum(item.duration_ms for item in node_results),
                    nodeids=case.required_nodeids,
                    failure_reasons=failure_reasons,
                )
            )

        total = len(case_results)
        passed = sum(item.status == "pass" for item in case_results)
        failed = sum(item.status == "fail" for item in case_results)
        skipped = sum(item.status == "skipped" for item in case_results)

        if failed:
            decision = "fail"
        elif skipped:
            decision = "inconclusive"
        else:
            decision = "pass"

        summary = StatefulFaultVerificationSummary(
            decision=decision,
            total_cases=total,
            passed_cases=passed,
            failed_cases=failed,
            skipped_cases=skipped,
            pass_rate=(passed / total) if total else 0.0,
            failed_case_ids=tuple(
                item.case_id for item in case_results if item.status == "fail"
            ),
            skipped_case_ids=tuple(
                item.case_id for item in case_results if item.status == "skipped"
            ),
        )

        return StatefulFaultVerificationReport(
            generated_at=datetime.now(timezone.utc),
            suite_id=suite.suite_id,
            suite_version=suite.suite_version,
            runner_version=runner_version,
            graph_version=graph_version,
            checkpoint_schema_version=checkpoint_schema_version,
            state_schema_version=state_schema_version,
            pytest_exit_code=observations.pytest_exit_code,
            summary=summary,
            cases=tuple(case_results),
        )
