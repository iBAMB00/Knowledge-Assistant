from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


StatefulFaultCategory = Literal[
    "graph",
    "checkpoint",
    "recovery",
    "hitl",
    "cancellation",
    "isolation",
    "compatibility",
    "runtime_parity",
    "production_api",
]
StatefulFaultCaseStatus = Literal["pass", "fail", "skipped"]
StatefulFaultDecision = Literal["pass", "fail", "inconclusive"]


class StatefulFaultCaseDefinition(BaseModel):
    """A12 故障验证套件中的一个可审计 Case。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(min_length=1, max_length=100)
    category: StatefulFaultCategory
    description: str = Field(min_length=1, max_length=500)
    required_nodeids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_nodeids(self) -> "StatefulFaultCaseDefinition":
        normalized = tuple(nodeid.strip() for nodeid in self.required_nodeids)
        if any(not nodeid for nodeid in normalized):
            raise ValueError("required_nodeids cannot contain empty values")
        if len(set(normalized)) != len(normalized):
            raise ValueError("required_nodeids must be unique within a case")
        return self


class StatefulFaultSuiteDefinition(BaseModel):
    """版本化 Stateful Fault Verification 套件。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    suite_id: str = Field(min_length=1, max_length=100)
    suite_version: str = Field(min_length=1, max_length=50)
    cases: tuple[StatefulFaultCaseDefinition, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_cases(self) -> "StatefulFaultSuiteDefinition":
        case_ids = [case.case_id for case in self.cases]
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("stateful fault case_id must be unique")
        return self


class StatefulFaultNodeObservation(BaseModel):
    """一个 pytest nodeid 的真实执行结果。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    nodeid: str = Field(min_length=1, max_length=300)
    status: StatefulFaultCaseStatus
    duration_ms: float = Field(ge=0)
    failure_message: str | None = Field(default=None, max_length=2000)


class StatefulFaultObservationSet(BaseModel):
    """A12 probe runner 产生的原始观察值。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    suite_id: str = Field(min_length=1, max_length=100)
    suite_version: str = Field(min_length=1, max_length=50)
    generated_at: datetime
    pytest_exit_code: int
    observations: tuple[StatefulFaultNodeObservation, ...]


class StatefulFaultCaseResult(BaseModel):
    """聚合到业务 Case 维度后的验证结果。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(min_length=1, max_length=100)
    category: StatefulFaultCategory
    description: str = Field(min_length=1, max_length=500)
    status: StatefulFaultCaseStatus
    duration_ms: float = Field(ge=0)
    nodeids: tuple[str, ...]
    failure_reasons: tuple[str, ...] = ()


class StatefulFaultVerificationSummary(BaseModel):
    """A12 Stateful Runtime 可靠性摘要。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    decision: StatefulFaultDecision
    total_cases: int = Field(ge=0)
    passed_cases: int = Field(ge=0)
    failed_cases: int = Field(ge=0)
    skipped_cases: int = Field(ge=0)
    pass_rate: float = Field(ge=0, le=1)
    failed_case_ids: tuple[str, ...] = ()
    skipped_case_ids: tuple[str, ...] = ()


class StatefulFaultVerificationReport(BaseModel):
    """v2.3 A12 可持久化 Fault / Recovery 验证证据。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    generated_at: datetime
    suite_id: str = Field(min_length=1, max_length=100)
    suite_version: str = Field(min_length=1, max_length=50)
    runner_version: str = Field(min_length=1, max_length=50)
    graph_version: str = Field(min_length=1, max_length=50)
    checkpoint_schema_version: str = Field(min_length=1, max_length=50)
    state_schema_version: str = Field(min_length=1, max_length=50)
    pytest_exit_code: int
    summary: StatefulFaultVerificationSummary
    cases: tuple[StatefulFaultCaseResult, ...]
