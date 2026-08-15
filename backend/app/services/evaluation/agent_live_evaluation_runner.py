from datetime import datetime, timezone
import time

from sqlalchemy.orm import Session

from app.agent.context import ToolExecutionContext
from app.agent.native_agent import AgentLoopError
from app.schemas.agent_evaluation import (
    AgentEvaluationDataset,
    AgentEvaluationObservation,
    AgentEvaluationObservationSet,
)
from app.services.agent_execution_service import AgentExecutionService
from app.services.evaluation.agent_dataset_binder import AgentEvaluationDatasetBinder
from app.services.evaluation.agent_observation_collector import (
    AgentEvaluationObservationCollector,
)
from app.services.knowledge_base_access_policy import KnowledgeBaseAccessPolicy


class AgentLiveEvaluationRunner:
    """
    D2 真实 Agent Observation Runner。

    使用与生产 /agent/chat 相同的 AgentExecutionService 执行 Dataset，
    但通过可选进程内 Observer 捕获 Eval 所需的 Tool 请求事实。

    D2 不猜测 answerable / grounded / sources / tokens / cost；这些字段在
    没有可靠观测来源时保持 None/空值，留给后续 D3/E/AgentOps。
    """

    RUNNER_VERSION = "1.0.0"

    def __init__(
        self,
        *,
        execution_service: AgentExecutionService,
        access_policy: KnowledgeBaseAccessPolicy,
    ) -> None:
        self.execution_service = execution_service
        self.access_policy = access_policy

    def run_dataset(
        self,
        *,
        db: Session,
        dataset: AgentEvaluationDataset,
        context: ToolExecutionContext,
    ) -> AgentEvaluationObservationSet:
        AgentEvaluationDatasetBinder.ensure_live_ready(dataset)
        self._validate_scope(db=db, context=context)

        observations = [
            self._run_case(
                db=db,
                dataset=dataset,
                base_context=context,
                case_id=case.case_id,
                query=case.query,
            )
            for case in dataset.cases
        ]

        return AgentEvaluationObservationSet(
            dataset_id=dataset.dataset_id,
            dataset_version=dataset.dataset_version,
            runner_version=self.RUNNER_VERSION,
            generated_at=datetime.now(timezone.utc),
            observations=observations,
        )

    def _run_case(
        self,
        *,
        db: Session,
        dataset: AgentEvaluationDataset,
        base_context: ToolExecutionContext,
        case_id: str,
        query: str,
    ) -> AgentEvaluationObservation:
        collector = AgentEvaluationObservationCollector()
        context = base_context.model_copy(
            update={
                "request_id": self._build_request_id(
                    dataset_version=dataset.dataset_version,
                    case_id=case_id,
                ),
                "agent_run_id": None,
            }
        )
        started_at = time.perf_counter()
        run_succeeded = False
        run_error_type: str | None = None

        try:
            self.execution_service.run(
                db=db,
                context=context,
                message=query,
                observer=collector,
            )
            run_succeeded = True
        except AgentLoopError as exc:
            run_error_type = exc.code
        except Exception as exc:
            run_error_type = type(exc).__name__

        latency_ms = max(
            0.0,
            (time.perf_counter() - started_at) * 1000,
        )

        return AgentEvaluationObservation(
            case_id=case_id,
            run_succeeded=run_succeeded,
            run_error_type=run_error_type,
            answerable=None,
            grounded=None,
            tool_calls=collector.build_tool_calls(),
            observed_sources=[],
            latency_ms=latency_ms,
            input_tokens=None,
            output_tokens=None,
            cost=None,
        )

    def _validate_scope(
        self,
        *,
        db: Session,
        context: ToolExecutionContext,
    ) -> None:
        self.access_policy.get_accessible_knowledge_base(
            db=db,
            knowledge_base_id=context.knowledge_base_id,
            user=context.to_access_principal(),
        )

    @staticmethod
    def _build_request_id(
        *,
        dataset_version: str,
        case_id: str,
    ) -> str:
        raw = f"agent-eval:{dataset_version}:{case_id}"
        return raw[:128]
