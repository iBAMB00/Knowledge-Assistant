import time
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.agent.context import ToolExecutionContext
from app.agent.native_agent import AgentLoopError
from app.agent.version_snapshot import AgentEvaluationVersionContext
from app.constants.agent_evaluation_runtime import (
    NATIVE_LIVE_EVALUATION_RUNNER_VERSION,
)
from app.schemas.agent_evaluation import (
    AgentEvaluationCase,
    AgentEvaluationDataset,
    AgentEvaluationObservation,
    AgentEvaluationObservationSet,
)
from app.services.agent_execution_service import AgentExecutionService
from app.services.evaluation.agent_dataset_binder import (
    AgentEvaluationDatasetBinder,
)
from app.services.evaluation.agent_evidence_loader import (
    AgentEvaluationEvidenceLoader,
)
from app.services.evaluation.agent_observation_collector import (
    AgentEvaluationObservationCollector,
)
from app.services.evaluation.groundedness_judge import (
    GroundednessJudge,
    GroundednessJudgeError,
)
from app.services.knowledge_base_access_policy import KnowledgeBaseAccessPolicy


class AgentLiveEvaluationRunner:
    """
    真实 Agent Observation Runner。

    使用与生产 /agent/chat 相同的 AgentExecutionService 执行 Dataset，
    通过可选进程内 Observer 捕获 Eval 所需的 Tool 请求事实。

    D3.2 起，标记 evaluate_groundedness 的 Case 会在 Agent Run 完成后，
    通过 Eval-only Evidence Loader 临时恢复证据正文并交给独立 Judge。
    最终回答与证据正文都不会写入 Observation / AgentRun / SSE。
    """

    RUNNER_VERSION = NATIVE_LIVE_EVALUATION_RUNNER_VERSION

    def __init__(
        self,
        *,
        execution_service: AgentExecutionService,
        access_policy: KnowledgeBaseAccessPolicy,
        groundedness_judge: GroundednessJudge | None = None,
        evidence_loader: AgentEvaluationEvidenceLoader | None = None,
        evaluator_version: str | None = None,
    ) -> None:
        self.execution_service = execution_service
        self.access_policy = access_policy
        self.groundedness_judge = groundedness_judge
        self.evidence_loader = evidence_loader
        self.evaluator_version = (
            evaluator_version.strip()
            if evaluator_version is not None
            else None
        )
        if evaluator_version is not None and not self.evaluator_version:
            raise ValueError("evaluator_version cannot be empty")

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
                case=case,
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
        case: AgentEvaluationCase,
    ) -> AgentEvaluationObservation:
        collector = AgentEvaluationObservationCollector()
        context = base_context.model_copy(
            update={
                "request_id": self._build_request_id(
                    dataset_version=dataset.dataset_version,
                    case_id=case.case_id,
                ),
                "agent_run_id": None,
            }
        )
        started_at = time.perf_counter()
        run_succeeded = False
        run_error_type: str | None = None
        final_answer: str | None = None

        try:
            evaluation_version = self._build_evaluation_version(
                dataset_version=dataset.dataset_version,
            )
            if evaluation_version is None:
                result = self.execution_service.run(
                    db=db,
                    context=context,
                    message=case.query,
                    observer=collector,
                )
            else:
                result = self.execution_service.run(
                    db=db,
                    context=context,
                    message=case.query,
                    observer=collector,
                    evaluation_version=evaluation_version,
                )
            final_answer = result.answer
            run_succeeded = True
        except AgentLoopError as exc:
            run_error_type = exc.code
        except Exception as exc:
            run_error_type = type(exc).__name__

        # 保持历史 latency_ms 语义：只统计 Agent Run，不把 Eval Judge 延迟混入。
        latency_ms = max(
            0.0,
            (time.perf_counter() - started_at) * 1000,
        )

        grounded: bool | None = None
        grounded_score: float | None = None
        grounded_judge_version: str | None = None
        grounded_judge_error_type: str | None = None

        if case.evaluate_groundedness:
            (
                grounded,
                grounded_score,
                grounded_judge_version,
                grounded_judge_error_type,
            ) = self._judge_groundedness(
                db=db,
                context=context,
                question=case.query,
                final_answer=final_answer,
                retrieved_sources=collector.build_retrieved_sources(),
            )

        return AgentEvaluationObservation(
            case_id=case.case_id,
            run_succeeded=run_succeeded,
            run_error_type=run_error_type,
            answerable=None,
            grounded=grounded,
            grounded_score=grounded_score,
            grounded_judge_version=grounded_judge_version,
            grounded_judge_error_type=grounded_judge_error_type,
            tool_calls=collector.build_tool_calls(),
            retrieved_sources=collector.build_retrieved_sources(),
            observed_sources=collector.build_observed_sources(),
            latency_ms=latency_ms,
            input_tokens=None,
            output_tokens=None,
            cost=None,
        )

    def _judge_groundedness(
        self,
        *,
        db: Session,
        context: ToolExecutionContext,
        question: str,
        final_answer: str | None,
        retrieved_sources: list[str],
    ) -> tuple[bool | None, float | None, str | None, str | None]:
        """执行 Eval-only Groundedness Judge，并只返回安全 verdict 元数据。"""

        judge = self.groundedness_judge
        evidence_loader = self.evidence_loader

        if final_answer is None:
            return None, None, getattr(judge, "version", None), "run_incomplete"

        if judge is None or evidence_loader is None:
            return None, None, getattr(judge, "version", None), "judge_unconfigured"

        judge_version = judge.version
        evidence_result = evidence_loader.load(
            db=db,
            knowledge_base_id=context.knowledge_base_id,
            source_refs=retrieved_sources,
        )
        if evidence_result.missing_source_refs:
            return None, None, judge_version, "evidence_unavailable"

        try:
            result = judge.judge(
                question=question,
                answer=final_answer,
                evidence=evidence_result.evidence,
            )
        except GroundednessJudgeError as exc:
            return None, None, judge_version, exc.code
        except Exception as exc:
            return None, None, judge_version, type(exc).__name__

        return result.grounded, result.score, judge_version, None


    def _build_evaluation_version(
        self,
        *,
        dataset_version: str,
    ) -> AgentEvaluationVersionContext | None:
        """为 Eval Run 构建只含版本号的持久化上下文。"""

        if self.evaluator_version is None:
            return None

        return AgentEvaluationVersionContext(
            dataset_version=dataset_version,
            evaluator_version=self.evaluator_version,
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
