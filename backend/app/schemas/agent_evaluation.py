from __future__ import annotations

from collections import Counter
from datetime import datetime
from enum import Enum
from typing import Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    field_validator,
    model_validator,
)


class AgentEvaluationCaseCategory(str, Enum):
    """Agent Eval 第一版用例类别。"""

    NO_TOOL = "NO_TOOL"
    ONE_TOOL = "ONE_TOOL"
    MULTI_TOOL = "MULTI_TOOL"
    NO_ANSWER = "NO_ANSWER"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    TOOL_ERROR = "TOOL_ERROR"
    INJECTION = "INJECTION"
    LONG_RUNNING = "LONG_RUNNING"
    APPROVAL = "APPROVAL"


class AgentExpectedToolCall(BaseModel):
    """
    单次期望 Tool Call 标注。

    expected_arguments 采用“子集匹配”：只标注真正稳定、需要评估的参数，
    未标注字段不参与 Argument Accuracy，避免把模型可合理改写的 query
    文本强行做全量字符串相等比较。
    """

    model_config = ConfigDict(extra="forbid")

    tool_name: str = Field(min_length=1, max_length=100)
    expected_arguments: dict[str, Any] = Field(default_factory=dict)

    @field_validator("tool_name")
    @classmethod
    def normalize_tool_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("tool_name cannot be empty")
        return normalized


class AgentEvaluationCase(BaseModel):
    """单条 Agent 评估 Case。"""

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1, max_length=100)
    query: str = Field(min_length=1)
    category: AgentEvaluationCaseCategory
    expected_behavior: str = Field(min_length=1)
    allowed_tools: list[str] = Field(default_factory=list)
    forbidden_tools: list[str] = Field(default_factory=list)
    expected_sources: list[str] = Field(default_factory=list)
    expected_answerable: bool
    expected_tool_calls: list[AgentExpectedToolCall] = Field(
        default_factory=list
    )
    notes: str | None = None

    @field_validator("case_id", "query", "expected_behavior")
    @classmethod
    def normalize_required_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("value cannot be empty")
        return normalized

    @field_validator("notes")
    @classmethod
    def normalize_optional_text(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator(
        "allowed_tools",
        "forbidden_tools",
        "expected_sources",
    )
    @classmethod
    def normalize_unique_text_list(
        cls,
        values: list[str],
    ) -> list[str]:
        normalized = [value.strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("text list cannot contain empty values")
        if len(set(normalized)) != len(normalized):
            raise ValueError("text list cannot contain duplicates")
        return normalized

    @model_validator(mode="after")
    def validate_case_contract(self) -> Self:
        allowed = set(self.allowed_tools)
        forbidden = set(self.forbidden_tools)

        overlap = allowed & forbidden
        if overlap:
            raise ValueError(
                "allowed_tools and forbidden_tools cannot overlap: "
                f"{sorted(overlap)}"
            )

        expected_names = [
            tool_call.tool_name
            for tool_call in self.expected_tool_calls
        ]
        unexpected_expected_tools = set(expected_names) - allowed
        if unexpected_expected_tools:
            raise ValueError(
                "expected_tool_calls must be included in allowed_tools: "
                f"{sorted(unexpected_expected_tools)}"
            )

        if self.category == AgentEvaluationCaseCategory.NO_TOOL:
            if self.expected_tool_calls:
                raise ValueError(
                    "NO_TOOL case cannot define expected_tool_calls"
                )
            if self.allowed_tools:
                raise ValueError(
                    "NO_TOOL case cannot allow tool calls"
                )

        if self.category == AgentEvaluationCaseCategory.ONE_TOOL:
            if len(self.expected_tool_calls) != 1:
                raise ValueError(
                    "ONE_TOOL case must define exactly one "
                    "expected_tool_call"
                )

        if self.category == AgentEvaluationCaseCategory.MULTI_TOOL:
            if len(self.expected_tool_calls) < 2:
                raise ValueError(
                    "MULTI_TOOL case must define at least two "
                    "expected_tool_calls"
                )

        return self


class AgentEvaluationDataset(BaseModel):
    """可版本化 Agent Eval Dataset。"""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"]
    dataset_id: str = Field(min_length=1, max_length=100)
    dataset_version: str = Field(min_length=1, max_length=50)
    description: str = Field(min_length=1)
    cases: list[AgentEvaluationCase] = Field(min_length=1)

    @field_validator("dataset_id", "dataset_version", "description")
    @classmethod
    def normalize_dataset_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("value cannot be empty")
        return normalized

    @model_validator(mode="after")
    def validate_dataset_contract(self) -> Self:
        case_ids = [case.case_id for case in self.cases]
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("case_id cannot contain duplicates")
        return self


class AgentEvaluationDatasetReference(BaseModel):
    """写入报告的数据集不可变来源快照。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str
    dataset_id: str
    dataset_version: str
    source_path: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    total_cases: PositiveInt


class AgentObservedToolCall(BaseModel):
    """一次 Agent Run 中可用于 Eval 的 Tool 调用观察值。"""

    model_config = ConfigDict(extra="forbid")

    tool_name: str = Field(min_length=1, max_length=100)
    arguments: dict[str, Any] = Field(default_factory=dict)
    error_code: str | None = None

    @field_validator("tool_name")
    @classmethod
    def normalize_tool_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("tool_name cannot be empty")
        return normalized


class AgentEvaluationObservation(BaseModel):
    """
    单条 Case 的运行观察值。

    answerable / grounded 暂时允许为空：D1 只负责确定性聚合，
    后续 D2/D3 可由真实 Agent Runner、人工标注或 Judge 补充。
    """

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1, max_length=100)
    run_succeeded: bool
    answerable: bool | None = None
    grounded: bool | None = None
    tool_calls: list[AgentObservedToolCall] = Field(default_factory=list)
    observed_sources: list[str] = Field(default_factory=list)
    latency_ms: NonNegativeFloat
    input_tokens: NonNegativeInt | None = None
    output_tokens: NonNegativeInt | None = None
    cost: NonNegativeFloat | None = None

    @field_validator("case_id")
    @classmethod
    def normalize_case_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("case_id cannot be empty")
        return normalized

    @field_validator("observed_sources")
    @classmethod
    def normalize_sources(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("observed_sources cannot contain empty values")
        if len(set(normalized)) != len(normalized):
            raise ValueError("observed_sources cannot contain duplicates")
        return normalized


class AgentEvaluationObservationSet(BaseModel):
    """与某个 Dataset 版本对应的一批 Agent 运行观察值。"""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str = Field(min_length=1, max_length=100)
    dataset_version: str = Field(min_length=1, max_length=50)
    observations: list[AgentEvaluationObservation] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_observations(self) -> Self:
        case_ids = [observation.case_id for observation in self.observations]
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("observation case_id cannot contain duplicates")
        return self


class AgentEvaluationCaseResult(BaseModel):
    """单条 Agent Eval Case 的确定性结果。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    category: AgentEvaluationCaseCategory
    task_success: bool
    tool_selection_pass: bool
    tool_argument_accuracy: float | None = Field(default=None, ge=0, le=1)
    unnecessary_tool_call_rate: float = Field(ge=0, le=1)
    unauthorized_tool_call_count: NonNegativeInt
    answerability_match: bool | None = None
    grounded_answer: bool | None = None
    citation_correctness: float | None = Field(default=None, ge=0, le=1)
    tool_call_count: NonNegativeInt
    latency_ms: NonNegativeFloat
    input_tokens: NonNegativeInt | None = None
    output_tokens: NonNegativeInt | None = None
    cost: NonNegativeFloat | None = None


class AgentEvaluationSummary(BaseModel):
    """Agent Eval 1.0 聚合指标。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    total_cases: PositiveInt
    task_success_rate: float = Field(ge=0, le=1)
    tool_selection_accuracy: float = Field(ge=0, le=1)
    tool_argument_accuracy: float | None = Field(default=None, ge=0, le=1)
    unnecessary_tool_call_rate: float = Field(ge=0, le=1)
    unauthorized_tool_call_count: NonNegativeInt
    grounded_answer_rate: float | None = Field(default=None, ge=0, le=1)
    citation_correctness: float | None = Field(default=None, ge=0, le=1)
    average_tool_calls: NonNegativeFloat
    average_latency_ms: NonNegativeFloat
    total_input_tokens: NonNegativeInt | None = None
    total_output_tokens: NonNegativeInt | None = None
    total_cost: NonNegativeFloat | None = None


class AgentEvaluationReport(BaseModel):
    """Agent Eval 1.0 可持久化报告。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    generated_at: datetime
    evaluator_version: str = Field(min_length=1, max_length=50)
    dataset: AgentEvaluationDatasetReference
    summary: AgentEvaluationSummary
    cases: list[AgentEvaluationCaseResult] = Field(min_length=1)
