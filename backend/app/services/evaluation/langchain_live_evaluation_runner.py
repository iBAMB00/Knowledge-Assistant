"""LangChain Candidate 对现有 Agent Eval Dataset 的真实运行桥接。"""

import time
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.agent.context import ToolExecutionContext
from app.agent.frameworks.langchain.runner import (
    LangChainAgentError,
    LangChainSingleAgentRunner,
)
from app.schemas.agent_evaluation import (
    AgentEvaluationCase,
    AgentEvaluationDataset,
    AgentEvaluationObservation,
    AgentEvaluationObservationSet,
)
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


class LangChainLiveEvaluationRunner:
    """
    v2.1-A3 LangChain Candidate Live Eval Runner。

    与 Native AgentLiveEvaluationRunner 共用：
    - AgentEvaluationDataset；
    - AgentEvaluationObservationCollector；
    - Groundedness Judge / Evidence Loader；
    - AgentEvaluator。

    A4 起 Candidate 已补齐 max_tool_calls / repeated_tool_call /
    operation-boundary timeout 三类 Native Runtime Guard；当前仍刻意不接
    AgentRun 生命周期持久化与生产 API，先保持 Candidate Eval 隔离。
    """

    RUNNER_VERSION = f"langchain-v1:{LangChainSingleAgentRunner.RUNNER_VERSION}"

    def __init__(
        self,
        *,
        agent_runner: LangChainSingleAgentRunner,
        access_policy: KnowledgeBaseAccessPolicy,
        groundedness_judge: GroundednessJudge | None = None,
        evidence_loader: AgentEvaluationEvidenceLoader | None = None,
    ) -> None:
        self.agent_runner = agent_runner
        self.access_policy = access_policy
        self.groundedness_judge = groundedness_judge
        self.evidence_loader = evidence_loader

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
            result = self.agent_runner.run(
                db=db,
                context=context,
                message=case.query,
                observer=collector,
            )
            final_answer = result.answer
            run_succeeded = True
        except LangChainAgentError as exc:
            run_error_type = exc.code
        except Exception as exc:
            run_error_type = type(exc).__name__

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
    def _build_request_id(*, dataset_version: str, case_id: str) -> str:
        raw = f"agent-eval-langchain:{dataset_version}:{case_id}"
        return raw[:128]
