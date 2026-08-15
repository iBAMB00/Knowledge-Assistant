import argparse
import json
from pathlib import Path

from app.services.evaluation.agent_case_loader import AgentEvaluationCaseLoader
from app.services.evaluation.agent_dataset_binder import AgentEvaluationDatasetBinder


DEFAULT_CASES_PATH = Path("evaluation/agent_cases.json")
DEFAULT_BOUND_CASES_PATH = Path("evaluation/generated/agent_cases.bound.json")
DEFAULT_MANIFEST_PATH = Path("evaluation/generated/agent_eval_fixture.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare idempotent DB fixtures for Agent Live Eval and bind "
            "environment-specific IDs into a generated dataset."
        )
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument(
        "--bound-cases-output",
        type=Path,
        default=DEFAULT_BOUND_CASES_PATH,
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
    )
    return parser.parse_args()


def _ensure_safe_environment(app_environment: str) -> None:
    """Eval Fixture 只允许写入非生产环境数据库。"""

    if app_environment == "production":
        raise RuntimeError(
            "agent evaluation fixtures cannot be prepared in production"
        )


def main() -> int:
    args = parse_args()

    # 延迟导入 DB/Settings，保证 --help 不初始化应用运行时。
    from app.core.config import get_settings
    from app.core.database import SessionLocal
    from app.repositories.document_repository import DocumentRepository
    from app.repositories.knowledge_base_repository import KnowledgeBaseRepository
    from app.repositories.user_repository import UserRepository
    from app.services.evaluation.agent_eval_fixture_service import (
        AgentEvaluationFixtureService,
    )

    _ensure_safe_environment(get_settings().app_environment)

    loader = AgentEvaluationCaseLoader()
    dataset = loader.load_dataset(args.cases)

    with SessionLocal() as db:
        manifest = AgentEvaluationFixtureService(
            user_repository=UserRepository(),
            knowledge_base_repository=KnowledgeBaseRepository(),
            document_repository=DocumentRepository(),
        ).prepare(db=db)

    bound_dataset = AgentEvaluationDatasetBinder.bind(
        dataset=dataset,
        manifest=manifest,
    )
    AgentEvaluationDatasetBinder.ensure_live_ready(bound_dataset)

    args.bound_cases_output.parent.mkdir(parents=True, exist_ok=True)
    args.bound_cases_output.write_text(
        bound_dataset.model_dump_json(indent=2),
        encoding="utf-8",
    )

    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(
        manifest.model_dump_json(indent=2),
        encoding="utf-8",
    )

    powershell_command = (
        "python -m scripts.run_agent_live_evaluation "
        f"--cases {args.bound_cases_output} "
        f"--fixture-manifest {args.manifest_output}"
    )

    print(
        json.dumps(
            {
                "fixture_version": manifest.fixture_version,
                "primary_user_id": manifest.primary_user_id,
                "primary_knowledge_base_id": manifest.primary_knowledge_base_id,
                "primary_document_id": manifest.primary_document_id,
                "cross_user_document_id": manifest.cross_user_document_id,
                "missing_processing_job_id": manifest.missing_processing_job_id,
                "bound_cases_output": str(args.bound_cases_output),
                "manifest_output": str(args.manifest_output),
                "powershell_command": powershell_command,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
