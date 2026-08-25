from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class EvaluationDomain(str, Enum):
    """Eval 2.0 统一评估领域。"""

    AGENT = "agent"
    RETRIEVAL = "retrieval"
    STATEFUL = "stateful"


class EvaluationMetricSource(str, Enum):
    """指标结论来源，避免把 Judge / Human 与确定性指标混为一谈。"""

    DETERMINISTIC = "deterministic"
    LLM_JUDGE = "llm_judge"
    HUMAN = "human"


class EvaluationMetricDirection(str, Enum):
    """指标优化方向。"""

    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"
    TARGET_ZERO = "target_zero"


class EvaluationMetric(BaseModel):
    """Eval 2.0 单项指标。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    metric_id: str = Field(min_length=1, max_length=100)
    value: float | int | None = None
    unit: str | None = Field(default=None, max_length=32)
    source: EvaluationMetricSource
    direction: EvaluationMetricDirection
    applicable: bool = True


class EvaluationDatasetIdentity(BaseModel):
    """不同旧 Eval 数据集统一后的不可变身份。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str = Field(min_length=1, max_length=100)
    dataset_version: str = Field(min_length=1, max_length=50)
    source_path: str | None = None
    source_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    total_cases: int = Field(ge=0)


class EvaluationVersionIdentity(BaseModel):
    """被评估对象和 Evaluator 的版本身份。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evaluator_version: str | None = Field(default=None, max_length=50)
    code_version: str | None = Field(default=None, max_length=100)
    agent_version: str | None = Field(default=None, max_length=64)
    prompt_id: str | None = Field(default=None, max_length=128)
    prompt_version: str | None = Field(default=None, max_length=64)
    retrieval_variant_id: str | None = Field(default=None, max_length=100)
    graph_version: str | None = Field(default=None, max_length=50)


class EvaluationCaseResultV2(BaseModel):
    """统一 Case 结果；原始领域报告仍然保留，不做有损替代。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(min_length=1, max_length=100)
    passed: bool | None = None
    trace_id: str | None = Field(default=None, max_length=128)
    provider_trace_id: str | None = Field(default=None, max_length=128)
    agent_run_id: int | str | None = None
    metrics: tuple[EvaluationMetric, ...] = ()


class EvaluationRunV2(BaseModel):
    """
    Eval 2.0 统一只读视图。

    它是已有 Retrieval / Agent / Stateful 报告之上的兼容层，
    不要求 v1 报告立即迁移或废弃。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["2.0"] = "2.0"
    domain: EvaluationDomain
    generated_at: datetime
    dataset: EvaluationDatasetIdentity
    version: EvaluationVersionIdentity
    summary_metrics: tuple[EvaluationMetric, ...]
    cases: tuple[EvaluationCaseResultV2, ...]
