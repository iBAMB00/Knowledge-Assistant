from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


MCPReleaseDecision = Literal["pass", "fail", "inconclusive"]


class MCPLiveVerificationReport(BaseModel):
    """一次真实 MCP stdio 验证留下的可持久化发布证据。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    generated_at: datetime
    verifier_version: str = Field(min_length=1, max_length=50)
    sdk_version: str = Field(min_length=1, max_length=50)
    transport: str = Field(min_length=1, max_length=32)
    server_id: str = Field(min_length=1, max_length=32)
    remote_tool_name: str = Field(min_length=1, max_length=128)
    exposed_tool_name: str = Field(min_length=1, max_length=128)
    tool_contract_version: str = Field(min_length=1, max_length=64)
    mcp_toolset_version: str = Field(min_length=1, max_length=64)
    discovery_succeeded: bool
    dispatch_succeeded: bool
    structured_output_succeeded: bool
    result_echo: str | None = Field(default=None, max_length=500)
    decision: MCPReleaseDecision
    failure_reason: str | None = Field(default=None, max_length=300)


class MCPReleaseGateResult(BaseModel):
    """v2.2 MCP Integration 的最终 Release Gate 结果。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    generated_at: datetime
    decision: MCPReleaseDecision
    release_ready: bool
    live_verification_decision: MCPReleaseDecision
    runtime_comparison_decision: MCPReleaseDecision
    mcp_tool_name: str
    toolset_version: str | None = None
    dataset_id: str
    dataset_version: str
    evaluator_version: str
    baseline_runner_version: str
    candidate_runner_version: str
    reasons: list[str] = Field(default_factory=list)
