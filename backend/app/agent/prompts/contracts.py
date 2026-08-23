"""Agent Runtime 共用的框架无关 Prompt 契约。"""

from pydantic import BaseModel, ConfigDict, Field


class PromptTemplateContract(BaseModel):
    """与具体 Agent 框架无关的版本化 Prompt 模板定义。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    prompt_id: str = Field(
        min_length=1,
        max_length=80,
        pattern=r"^[a-z0-9][a-z0-9._-]*$",
    )
    version: str = Field(min_length=1, max_length=64)
    template: str = Field(min_length=1)
    variables: tuple[str, ...] = ()


class RenderedPrompt(BaseModel):
    """Prompt 渲染结果，同时保留诊断与版本快照需要的身份信息。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    prompt_id: str = Field(min_length=1, max_length=80)
    version: str = Field(min_length=1, max_length=64)
    content: str = Field(min_length=1)
