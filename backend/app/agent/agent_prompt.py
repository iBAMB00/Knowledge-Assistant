"""Agent Runtime 共用 Prompt Catalog 与兼容入口。"""

from app.agent.prompts import (
    PromptRenderer,
    PromptTemplateContract,
    RenderedPrompt,
)


_BASE_AGENT_SYSTEM_PROMPT_TEXT = (
    "你是一个企业私有知识助手，"
    "请基于已知信息准确、简洁地"
    "回答用户问题。"
)

_AGENT_TOOL_CALLING_RULES = (
    " 对于能力介绍、身份说明、简单寒暄，以及询问你能做什么、"
    "有哪些能力或有哪些工具的元问题，直接根据系统说明和当前已提供"
    "的 Tool 定义回答，不要为了确认自身能力调用任何业务 Tool。"
    "只有当回答用户的业务问题确实需要读取私有数据、检索知识或查询"
    "业务状态时才调用 Tool。"
    " 当 search_knowledge 返回一个或多个 source_ref 后，如果最终"
    "回答使用了这些检索结果中的任何知识事实，必须在对应事实附近至少"
    "引用一个实际使用的 source_ref，格式严格为 [source:<source_ref>]。"
    "不得使用 [1]、来源1、Markdown 链接或裸 doc:... 替代标准格式，"
    "也不得编造未由 Tool 返回的 source_ref。若检索结果与用户问题无关"
    "或不足以支持答案，应明确说明证据不足，并且不要为了满足格式而引用"
    "无关 source_ref。"
)

BASE_AGENT_SYSTEM_PROMPT = PromptTemplateContract(
    prompt_id="agent.base-system",
    version="1.0.0",
    template=_BASE_AGENT_SYSTEM_PROMPT_TEXT,
)

AGENT_TOOL_CALLING_SYSTEM_PROMPT = PromptTemplateContract(
    prompt_id="agent.tool-calling-system",
    version="1.0.0",
    template=_BASE_AGENT_SYSTEM_PROMPT_TEXT + _AGENT_TOOL_CALLING_RULES,
)

# 保留旧常量兼容现有版本快照与调用方。
AGENT_TOOL_CALLING_PROMPT_VERSION = AGENT_TOOL_CALLING_SYSTEM_PROMPT.version

_PROMPT_RENDERER = PromptRenderer()


def render_base_agent_system_prompt() -> RenderedPrompt:
    """渲染基础系统 Prompt，并保留 Prompt 身份与版本。"""

    return _PROMPT_RENDERER.render(BASE_AGENT_SYSTEM_PROMPT)


def render_agent_tool_calling_system_prompt() -> RenderedPrompt:
    """渲染三套 Runtime 共用的 Tool Calling Prompt Contract。"""

    return _PROMPT_RENDERER.render(AGENT_TOOL_CALLING_SYSTEM_PROMPT)


def build_base_agent_system_prompt() -> str:
    """兼容旧调用方，只返回基础系统 Prompt 文本。"""

    return render_base_agent_system_prompt().content


def build_agent_tool_calling_system_prompt() -> str:
    """兼容旧调用方，只返回 Tool Calling Prompt 文本。"""

    return render_agent_tool_calling_system_prompt().content
