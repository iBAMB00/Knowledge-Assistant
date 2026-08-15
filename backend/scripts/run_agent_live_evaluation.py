import argparse
import json
from pathlib import Path

from app.agent.context import ToolExecutionContext
from app.constants.user_role import UserRole
from app.services.evaluation.agent_case_loader import AgentEvaluationCaseLoader
from app.services.evaluation.agent_dataset_binder import AgentEvaluationDatasetBinder
from app.services.evaluation.agent_evaluator import AgentEvaluator


DEFAULT_CASES_PATH = Path("evaluation/agent_cases.json")
DEFAULT_OBSERVATIONS_PATH = Path(
    "evaluation/reports/agent_live_observations_v1.json"
)
DEFAULT_REPORT_PATH = Path("evaluation/reports/agent_live_evaluation_v1.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the current Native Agent against Agent Eval cases, "
            "capture real Tool observations, and calculate deterministic metrics."
        )
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--user-id", type=int)
    parser.add_argument(
        "--role",
        choices=[role.value for role in UserRole],
        default=UserRole.USER.value,
    )
    parser.add_argument("--knowledge-base-id", type=int)
    parser.add_argument(
        "--fixture-manifest",
        type=Path,
        help=(
            "Use user/role/knowledge-base scope from a D2.5 fixture manifest. "
            "When provided, manual scope arguments are not required."
        ),
    )
    parser.add_argument(
        "--observations-output",
        type=Path,
        default=DEFAULT_OBSERVATIONS_PATH,
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=DEFAULT_REPORT_PATH,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # 延迟导入运行时依赖，保证 --help / 静态导入不会提前初始化 DB/LLM/Qdrant。
    from app.api.dependencies.agent import (
        get_agent_access_policy,
        get_agent_execution_service,
    )
    from app.core.database import SessionLocal
    from app.services.evaluation.agent_live_evaluation_runner import (
        AgentLiveEvaluationRunner,
    )

    loader = AgentEvaluationCaseLoader()
    dataset = loader.load_dataset(args.cases)
    dataset_reference = loader.build_reference(dataset, args.cases)

    if args.fixture_manifest is not None:
        manifest = AgentEvaluationDatasetBinder.load_manifest(
            args.fixture_manifest
        )
        user_id = manifest.primary_user_id
        role = UserRole(manifest.primary_role)
        knowledge_base_id = manifest.primary_knowledge_base_id
    else:
        if args.user_id is None or args.knowledge_base_id is None:
            raise ValueError(
                "provide --fixture-manifest or both --user-id and "
                "--knowledge-base-id"
            )
        user_id = args.user_id
        role = UserRole(args.role)
        knowledge_base_id = args.knowledge_base_id

    context = ToolExecutionContext(
        user_id=user_id,
        role=role,
        knowledge_base_id=knowledge_base_id,
        request_id="agent-eval-bootstrap",
    )

    with SessionLocal() as db:
        observations = AgentLiveEvaluationRunner(
            execution_service=get_agent_execution_service(),
            access_policy=get_agent_access_policy(),
        ).run_dataset(
            db=db,
            dataset=dataset,
            context=context,
        )

    loader.validate_observation_coverage(dataset, observations)
    report = AgentEvaluator().evaluate(
        dataset=dataset,
        dataset_reference=dataset_reference,
        observations=observations,
    )

    args.observations_output.parent.mkdir(parents=True, exist_ok=True)
    args.observations_output.write_text(
        observations.model_dump_json(indent=2),
        encoding="utf-8",
    )

    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(
        report.model_dump_json(indent=2),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "dataset_id": dataset.dataset_id,
                "dataset_version": dataset.dataset_version,
                "runner_version": observations.runner_version,
                "observations_output": str(args.observations_output),
                "report_output": str(args.report_output),
                "task_success_rate": report.summary.task_success_rate,
                "tool_selection_accuracy": (
                    report.summary.tool_selection_accuracy
                ),
                "tool_execution_accuracy": (
                    report.summary.tool_execution_accuracy
                ),
                "tool_argument_accuracy": (
                    report.summary.tool_argument_accuracy
                ),
                "tool_policy_violation_count": (
                    report.summary.tool_policy_violation_count
                ),
                "average_latency_ms": report.summary.average_latency_ms,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
