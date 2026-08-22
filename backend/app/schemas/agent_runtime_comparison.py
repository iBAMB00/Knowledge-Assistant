from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.agent_evaluation import (
    AgentEvaluationCaseCategory,
    AgentEvaluationDatasetReference,
)


AgentRuntimeComparisonDecision = Literal[
    "pass",
    "fail",
    "inconclusive",
]
AgentRuntimeMetricStatus = Literal[
    "pass",
    "fail",
    "inconclusive",
    "informational",
]
AgentRuntimeMetricDirection = Literal[
    "higher_is_better",
    "lower_is_better",
    "informational",
]
GroundednessGateStatus = Literal[
    "pass",
    "fail",
    "inconclusive",
    "not_applicable",
]


class AgentRuntimeMetricCheck(BaseModel):
    """Native Baseline 与 Candidate 的单个聚合指标对比。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    metric: str = Field(min_length=1, max_length=100)
    direction: AgentRuntimeMetricDirection
    baseline_value: float | None = None
    candidate_value: float | None = None
    delta: float | None = None
    status: AgentRuntimeMetricStatus
    reason: str | None = Field(default=None, max_length=300)


class AgentRuntimeCaseComparison(BaseModel):
    """单条 Eval Case 的 Runtime 对比快照。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(min_length=1, max_length=100)
    category: AgentEvaluationCaseCategory
    regression_reasons: list[str] = Field(default_factory=list)
    improvement_reasons: list[str] = Field(default_factory=list)
    baseline_run_succeeded: bool
    candidate_run_succeeded: bool
    baseline_task_success: bool
    candidate_task_success: bool
    baseline_tool_call_count: int = Field(ge=0)
    candidate_tool_call_count: int = Field(ge=0)
    baseline_latency_ms: float = Field(ge=0)
    candidate_latency_ms: float = Field(ge=0)


class AgentRuntimeComparisonSummary(BaseModel):
    """Runtime 候选门禁的最终摘要。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    decision: AgentRuntimeComparisonDecision
    deterministic_gate_passed: bool
    groundedness_gate_status: GroundednessGateStatus
    failed_metrics: list[str] = Field(default_factory=list)
    inconclusive_metrics: list[str] = Field(default_factory=list)
    regression_case_ids: list[str] = Field(default_factory=list)
    improvement_case_ids: list[str] = Field(default_factory=list)
    task_success_rate_delta: float
    average_tool_calls_delta: float
    average_latency_ms_delta: float
    average_latency_ratio: float | None = None


class AgentRuntimeComparisonReport(BaseModel):
    """Native Baseline 与 Framework Candidate 的可持久化对比报告。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    generated_at: datetime
    dataset: AgentEvaluationDatasetReference
    evaluator_version: str = Field(min_length=1, max_length=50)
    baseline_runner_version: str = Field(min_length=1, max_length=50)
    candidate_runner_version: str = Field(min_length=1, max_length=50)
    toolset_version: str | None = Field(default=None, max_length=64)
    tool_names: list[str] = Field(default_factory=list)
    summary: AgentRuntimeComparisonSummary
    metric_checks: list[AgentRuntimeMetricCheck]
    case_comparisons: list[AgentRuntimeCaseComparison]
