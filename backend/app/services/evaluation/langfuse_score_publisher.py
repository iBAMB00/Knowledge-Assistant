"""Thin Eval 2.0 -> Langfuse score bridge for v2.5-A8."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

from app.agent.observability.provider import ObservabilityProvider
from app.schemas.evaluation_v2 import EvaluationRunV2


@dataclass(frozen=True, slots=True)
class LangfuseScorePublishSummary:
    attempted: int
    published: int
    skipped: int


class LangfuseEvaluationScorePublisher:
    """Publish existing local Eval facts without moving evaluator logic to Langfuse."""

    def __init__(self, provider: ObservabilityProvider) -> None:
        self.provider = provider

    def publish(self, run: EvaluationRunV2) -> LangfuseScorePublishSummary:
        attempted = 0
        published = 0
        skipped = 0

        if not self.provider.enabled:
            return LangfuseScorePublishSummary(
                attempted=0,
                published=0,
                skipped=len(run.cases),
            )

        for case in run.cases:
            trace_id = (case.provider_trace_id or "").strip()
            if not trace_id:
                skipped += 1
                continue

            comment = (
                f"dataset={run.dataset.dataset_id}:{run.dataset.dataset_version}; "
                f"case={case.case_id}; evaluator={run.version.evaluator_version or 'unknown'}"
            )
            if case.passed is not None:
                attempted += 1
                if self.provider.publish_trace_score(
                    provider_trace_id=trace_id,
                    name="eval.task_success",
                    value=1.0 if case.passed else 0.0,
                    data_type="BOOLEAN",
                    comment=comment,
                    score_id=self._score_id(
                        run=run,
                        trace_id=trace_id,
                        case_id=case.case_id,
                        metric_id="task_success",
                    ),
                ):
                    published += 1

            for metric in case.metrics:
                if not metric.applicable or metric.value is None:
                    continue
                # task_success is already published as BOOLEAN above.
                if metric.metric_id == "task_success":
                    continue
                attempted += 1
                if self.provider.publish_trace_score(
                    provider_trace_id=trace_id,
                    name=f"eval.{metric.metric_id}",
                    value=float(metric.value),
                    data_type="NUMERIC",
                    comment=comment,
                    score_id=self._score_id(
                        run=run,
                        trace_id=trace_id,
                        case_id=case.case_id,
                        metric_id=metric.metric_id,
                    ),
                ):
                    published += 1

        return LangfuseScorePublishSummary(
            attempted=attempted,
            published=published,
            skipped=skipped,
        )

    @staticmethod
    def _score_id(
        *,
        run: EvaluationRunV2,
        trace_id: str,
        case_id: str,
        metric_id: str,
    ) -> str:
        """Build a deterministic idempotency key for repeat score publishing."""

        material = "|".join(
            [
                trace_id,
                run.dataset.dataset_id,
                run.dataset.dataset_version,
                case_id,
                metric_id,
                run.version.evaluator_version or "unknown",
            ]
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()
