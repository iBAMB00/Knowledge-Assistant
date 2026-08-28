from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.schemas.evaluation_v2 import (
    EvaluationCaseResultV2,
    EvaluationDatasetIdentity,
    EvaluationDomain,
    EvaluationMetric,
    EvaluationMetricDirection,
    EvaluationMetricSource,
    EvaluationRunV2,
    EvaluationVersionIdentity,
)
from app.services.evaluation.langfuse_score_publisher import (
    LangfuseEvaluationScorePublisher,
)


@dataclass
class RecordingProvider:
    enabled: bool = True
    calls: list[dict[str, Any]] = field(default_factory=list)
    name: str = "recording"

    def publish_trace_score(self, **kwargs: Any) -> bool:
        self.calls.append(kwargs)
        return True

    def start_trace(self, **kwargs: Any):  # pragma: no cover - unused protocol surface
        raise AssertionError(kwargs)

    def flush(self) -> None:
        return None

    def shutdown(self) -> None:
        return None


def _run() -> EvaluationRunV2:
    return EvaluationRunV2(
        domain=EvaluationDomain.AGENT,
        generated_at=datetime.now(timezone.utc),
        dataset=EvaluationDatasetIdentity(
            dataset_id="agent-eval",
            dataset_version="2.0.0",
            total_cases=2,
        ),
        version=EvaluationVersionIdentity(evaluator_version="2.0.0"),
        summary_metrics=(),
        cases=(
            EvaluationCaseResultV2(
                case_id="case-1",
                passed=True,
                provider_trace_id="a" * 32,
                metrics=(
                    EvaluationMetric(
                        metric_id="task_success",
                        value=1,
                        source=EvaluationMetricSource.DETERMINISTIC,
                        direction=EvaluationMetricDirection.HIGHER_IS_BETTER,
                    ),
                    EvaluationMetric(
                        metric_id="tool_sequence_pass",
                        value=1,
                        source=EvaluationMetricSource.DETERMINISTIC,
                        direction=EvaluationMetricDirection.HIGHER_IS_BETTER,
                    ),
                ),
            ),
            EvaluationCaseResultV2(
                case_id="case-2",
                passed=False,
                provider_trace_id=None,
                metrics=(),
            ),
        ),
    )


def test_eval_v2_scores_are_published_to_existing_provider_trace() -> None:
    provider = RecordingProvider()
    summary = LangfuseEvaluationScorePublisher(provider).publish(_run())  # type: ignore[arg-type]

    assert summary.attempted == 2
    assert summary.published == 2
    assert summary.skipped == 1
    assert provider.calls[0]["provider_trace_id"] == "a" * 32
    assert provider.calls[0]["name"] == "eval.task_success"
    assert provider.calls[0]["data_type"] == "BOOLEAN"
    assert provider.calls[1]["name"] == "eval.tool_sequence_pass"


def test_disabled_provider_keeps_eval_local_without_error() -> None:
    provider = RecordingProvider(enabled=False)
    summary = LangfuseEvaluationScorePublisher(provider).publish(_run())  # type: ignore[arg-type]

    assert summary.attempted == 0
    assert summary.published == 0
    assert summary.skipped == 2
    assert provider.calls == []


def test_score_publish_ids_are_stable_across_repeated_publication() -> None:
    provider = RecordingProvider()
    publisher = LangfuseEvaluationScorePublisher(provider)  # type: ignore[arg-type]

    publisher.publish(_run())
    first_ids = [call["score_id"] for call in provider.calls]
    provider.calls.clear()
    publisher.publish(_run())
    second_ids = [call["score_id"] for call in provider.calls]

    assert first_ids == second_ids
    assert len(first_ids) == 2
    assert all(len(score_id) == 64 for score_id in first_ids)
