"""PromptOps identity contracts linking runtime traces with evaluation evidence."""

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.constants.agent_runtime import AgentRuntime


class AgentPromptReference(BaseModel):
    """Stable Prompt identity only; prompt content deliberately stays out."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    prompt_id: str = Field(min_length=1, max_length=128)
    prompt_version: str = Field(min_length=1, max_length=64)

    @field_validator("prompt_id", "prompt_version", mode="before")
    @classmethod
    def normalize_text(cls, value: object) -> object:
        if isinstance(value, str):
            normalized = value.strip()
            return normalized or None
        return value


class AgentTracePromptLink(BaseModel):
    """Safe link from one Agent run/trace to the Prompt version it used."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    trace_id: str = Field(min_length=1, max_length=128)
    provider_trace_id: str | None = Field(default=None, min_length=1, max_length=128)
    agent_run_id: int | str | None = None
    runtime: AgentRuntime
    agent_version: str = Field(min_length=1, max_length=64)
    prompt: AgentPromptReference

    @field_validator("trace_id", "provider_trace_id", "agent_version", mode="before")
    @classmethod
    def normalize_text(cls, value: object) -> object:
        if isinstance(value, str):
            normalized = value.strip()
            return normalized or None
        return value

    @field_validator("agent_run_id", mode="before")
    @classmethod
    def normalize_agent_run_id(cls, value: object) -> object:
        if isinstance(value, str):
            normalized = value.strip()
            if not normalized:
                raise ValueError("agent_run_id cannot be empty")
            return normalized
        return value
