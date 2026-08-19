from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.constants.agent_runtime import AgentRuntime


AgentRuntimeRole = Literal["baseline", "candidate"]
AgentFrameworkReleaseDecision = Literal[
    "pass",
    "fail",
    "inconclusive",
]


class AgentRuntimeCapabilityResponse(BaseModel):
    """单个 Agent Runtime 当前部署可见的安全能力摘要。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    runtime: AgentRuntime
    role: AgentRuntimeRole
    enabled: bool
    supports_sync: bool
    supports_stream: bool
    implementation_version: str = Field(min_length=1, max_length=64)


class AgentRuntimeStatusResponse(BaseModel):
    """认证用户可查询的 Agent Runtime 能力状态。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    default_runtime: AgentRuntime
    runtimes: list[AgentRuntimeCapabilityResponse]


class AgentFrameworkReleaseGateResult(BaseModel):
    """v2.1 Framework Candidate 的可持久化 Release Gate 结果。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    decision: AgentFrameworkReleaseDecision
    release_ready: bool
    comparison_decision: str = Field(min_length=1, max_length=32)
    dataset_id: str = Field(min_length=1, max_length=100)
    dataset_version: str = Field(min_length=1, max_length=50)
    evaluator_version: str = Field(min_length=1, max_length=50)
    expected_evaluator_version: str = Field(min_length=1, max_length=50)
    baseline_runner_version: str = Field(min_length=1, max_length=50)
    expected_baseline_runner_version: str = Field(min_length=1, max_length=50)
    candidate_runner_version: str = Field(min_length=1, max_length=50)
    expected_candidate_runner_version: str = Field(min_length=1, max_length=50)
    candidate_feature_gate_enabled: bool
    reasons: list[str] = Field(default_factory=list)
