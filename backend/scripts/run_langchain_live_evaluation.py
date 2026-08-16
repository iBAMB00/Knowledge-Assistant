"""使用 LangChain Candidate Runner 运行现有 Agent Eval Dataset。"""

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
    "evaluation/reports/langchain_live_observations_v1.json"
)
DEFAULT_REPORT_PATH = Path(
    "evaluation/reports/langchain_live_evaluation_v1.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the LangChain Candidate against the same Agent Eval cases "
            "and calculate the existing deterministic metrics."
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
    parser.add_argument("--fixture-manifest", type=Path)
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
    parser.add_argument(
        "--skip-groundedness-judge",
        action="store_true",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # 延迟导入重运行时依赖，保持 --help 与静态导入轻量。
    from app.agent.frameworks.langchain.model_adapter import LangChainModelAdapter
    from app.agent.frameworks.langchain.runner import LangChainSingleAgentRunner
    from app.api.dependencies.agent import get_agent_access_policy, get_agent_tools
    from app.core.config import get_settings
    from app.core.database import SessionLocal
    from app.repositories.document_chunk_repository import DocumentChunkRepository
    from app.services.evaluation.agent_evidence_loader import (
        AgentEvaluationEvidenceLoader,
    )
    from app.services.evaluation.groundedness_judge import (
        OpenAICompatibleGroundednessJudge,
    )
    from app.services.evaluation.langchain_live_evaluation_runner import (
        LangChainLiveEvaluationRunner,
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
        request_id="agent-eval-langchain-bootstrap",
    )

    settings = get_settings()
    groundedness_judge = (
        None
        if args.skip_groundedness_judge
        else OpenAICompatibleGroundednessJudge()
    )
    evidence_loader = AgentEvaluationEvidenceLoader(
        chunk_repository=DocumentChunkRepository(),
        parent_child_enabled=settings.parent_child_enabled,
    )
    candidate_runner = LangChainSingleAgentRunner(
        model=LangChainModelAdapter(settings).build(),
        tools=get_agent_tools(),
    )

    with SessionLocal() as db:
        observations = LangChainLiveEvaluationRunner(
            agent_runner=candidate_runner,
            access_policy=get_agent_access_policy(),
            groundedness_judge=groundedness_judge,
            evidence_loader=evidence_loader,
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
                "runtime": "langchain_candidate",
                "dataset_id": dataset.dataset_id,
                "dataset_version": dataset.dataset_version,
                "runner_version": observations.runner_version,
                "evaluator_version": report.evaluator_version,
                "observations_output": str(args.observations_output),
                "report_output": str(args.report_output),
                "task_success_rate": report.summary.task_success_rate,
                "tool_selection_accuracy": report.summary.tool_selection_accuracy,
                "tool_execution_accuracy": report.summary.tool_execution_accuracy,
                "tool_argument_accuracy": report.summary.tool_argument_accuracy,
                "tool_policy_violation_count": (
                    report.summary.tool_policy_violation_count
                ),
                "grounded_answer_rate": report.summary.grounded_answer_rate,
                "groundedness_coverage": report.summary.groundedness_coverage,
                "required_evidence_success_rate": (
                    report.summary.required_evidence_success_rate
                ),
                "required_citation_success_rate": (
                    report.summary.required_citation_success_rate
                ),
                "citation_correctness": report.summary.citation_correctness,
                "average_latency_ms": report.summary.average_latency_ms,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
