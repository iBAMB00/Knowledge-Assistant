from pydantic import BaseModel, ConfigDict, Field


class LLMToolCall(BaseModel):
    """模型请求执行的一次 Tool Call。"""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    arguments_json: str = Field(min_length=1)


class LLMToolResponse(BaseModel):
    """
    LLMService 对 Tool Calling 响应的最小稳定表示。

    B1 阶段只保留模型文本与原始 Tool Call，
    不在这里解析 arguments，也不执行 Tool。
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    content: str | None = None
    tool_calls: list[LLMToolCall] = Field(default_factory=list)
