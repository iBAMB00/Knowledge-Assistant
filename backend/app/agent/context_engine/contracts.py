"""Agent Runtime 共用的框架无关上下文契约。"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class AgentContextRole(StrEnum):
    """模型可见消息的稳定角色。"""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class AgentContextSource(StrEnum):
    """上下文内容来自哪里，避免把 History / Knowledge / Memory 混为一类。"""

    SYSTEM_PROMPT = "system_prompt"
    CURRENT_MESSAGE = "current_message"
    CONVERSATION_HISTORY = "conversation_history"
    CONVERSATION_SUMMARY = "conversation_summary"
    KNOWLEDGE = "knowledge"
    TOOL = "tool"
    MEMORY = "memory"


class AgentContextItem(BaseModel):
    """一条进入 Agent 模型上下文的 provider-neutral 内容。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: AgentContextRole
    source: AgentContextSource
    content: str = Field(min_length=1)
    source_id: str | None = Field(default=None, min_length=1, max_length=160)
    source_version: str | None = Field(default=None, min_length=1, max_length=64)


class AgentContextBudget(BaseModel):
    """一次 Context Builder 预算选择的可解释结果。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_tokens: int = Field(gt=0)
    estimated_tokens: int = Field(ge=0)
    required_tokens: int = Field(ge=0)
    input_supporting_tokens: int = Field(ge=0)
    selected_supporting_tokens: int = Field(ge=0)
    input_supporting_items: int = Field(ge=0)
    selected_supporting_items: int = Field(ge=0)
    dropped_supporting_items: int = Field(ge=0)
    truncated: bool
    required_over_budget: bool
    estimator_version: str = Field(min_length=1, max_length=64)


class AgentContextUsage(BaseModel):
    """对外可安全暴露的本轮 Context 使用摘要。

    只表达 History / Summary / Memory 是否真正进入预算选择后的模型 Context，
    不暴露正文、Prompt、Token 明细或安全策略。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    history: bool = False
    summary: bool = False
    memory: bool = False


class AgentContext(BaseModel):
    """一次模型调用前已经按顺序组装好的稳定上下文。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    items: tuple[AgentContextItem, ...] = Field(min_length=2)
    budget: AgentContextBudget | None = None

    def items_from(self, source: AgentContextSource) -> tuple[AgentContextItem, ...]:
        """按来源读取上下文，供 Runtime Adapter 做最小序列化。"""

        return tuple(item for item in self.items if item.source == source)
