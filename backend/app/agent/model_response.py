from pydantic import BaseModel, ConfigDict, Field, model_validator


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
    LLMService 对 Tool Calling 响应的稳定内部表示。

    只表达模型返回的文本与 Tool Call，
    不承担参数解析、Tool 执行或 Agent Loop 控制。
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    content: str | None = None
    tool_calls: list[LLMToolCall] = Field(default_factory=list)


class LLMToolResult(BaseModel):
    """一次 Tool 执行后回填给模型的稳定结果消息。"""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    call_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    content_json: str = Field(min_length=1)


class LLMToolExchange(BaseModel):
    """
    一轮“模型 Tool Call -> Tool Result”交换。

    Agent Runtime 保存 provider-neutral 历史，
    具体 OpenAI-compatible message 序列化由 LLMService 负责。
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    response: LLMToolResponse
    tool_results: list[LLMToolResult]

    @model_validator(mode="after")
    def validate_tool_results_match_calls(self) -> "LLMToolExchange":
        """每个 Tool Call 必须且只能有一个对应 Tool Result。"""

        call_ids = [tool_call.id for tool_call in self.response.tool_calls]
        result_ids = [tool_result.call_id for tool_result in self.tool_results]

        if len(call_ids) != len(set(call_ids)):
            raise ValueError("duplicate tool call id in response")

        if len(result_ids) != len(set(result_ids)):
            raise ValueError("duplicate tool result call_id")

        if set(call_ids) != set(result_ids):
            raise ValueError("tool results do not match tool calls")

        return self
