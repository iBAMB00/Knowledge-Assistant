import pytest
from pydantic import ValidationError

from app.agent.agent_prompt import (
    AGENT_TOOL_CALLING_PROMPT_VERSION,
    AGENT_TOOL_CALLING_SYSTEM_PROMPT,
    BASE_AGENT_SYSTEM_PROMPT,
    build_agent_tool_calling_system_prompt,
    build_base_agent_system_prompt,
    render_agent_tool_calling_system_prompt,
    render_base_agent_system_prompt,
)
from app.agent.prompts import PromptRenderer, PromptTemplateContract
from app.services.llm_service import LLMService


def test_agent_prompt_catalog_has_stable_identity_and_version():
    assert BASE_AGENT_SYSTEM_PROMPT.prompt_id == "agent.base-system"
    assert BASE_AGENT_SYSTEM_PROMPT.version == "1.0.0"
    assert (
        AGENT_TOOL_CALLING_SYSTEM_PROMPT.prompt_id
        == "agent.tool-calling-system"
    )
    assert AGENT_TOOL_CALLING_SYSTEM_PROMPT.version == "1.1.0"
    assert (
        AGENT_TOOL_CALLING_PROMPT_VERSION
        == AGENT_TOOL_CALLING_SYSTEM_PROMPT.version
    )
    assert (
        LLMService.AGENT_PROMPT_VERSION
        == AGENT_TOOL_CALLING_SYSTEM_PROMPT.version
    )
    assert (
        LLMService.AGENT_PROMPT_ID
        == AGENT_TOOL_CALLING_SYSTEM_PROMPT.prompt_id
    )


def test_prompt_helpers_render_contract_without_changing_prompt_content():
    renderer = PromptRenderer()

    assert render_base_agent_system_prompt() == renderer.render(
        BASE_AGENT_SYSTEM_PROMPT
    )
    assert render_agent_tool_calling_system_prompt() == renderer.render(
        AGENT_TOOL_CALLING_SYSTEM_PROMPT
    )
    assert build_base_agent_system_prompt() == render_base_agent_system_prompt().content
    assert (
        build_agent_tool_calling_system_prompt()
        == render_agent_tool_calling_system_prompt().content
    )
    assert "不要为了确认自身能力调用任何业务 Tool" in (
        build_agent_tool_calling_system_prompt()
    )
    assert "[source:<source_ref>]" in build_agent_tool_calling_system_prompt()
    assert "企业知识冲突" in build_agent_tool_calling_system_prompt()
    assert "不得把 Memory 当作企业知识来源" in build_agent_tool_calling_system_prompt()


def test_prompt_renderer_requires_declared_variables_and_rejects_extra_values():
    contract = PromptTemplateContract(
        prompt_id="test.context",
        version="1.0.0",
        template="Question: {question}",
        variables=("question",),
    )
    renderer = PromptRenderer()

    rendered = renderer.render(contract, {"question": "Qdrant?"})
    assert rendered.prompt_id == "test.context"
    assert rendered.version == "1.0.0"
    assert rendered.content == "Question: Qdrant?"

    with pytest.raises(ValueError, match="missing prompt variables: question"):
        renderer.render(contract)

    with pytest.raises(ValueError, match="unexpected prompt variables: extra"):
        renderer.render(
            contract,
            {"question": "Qdrant?", "extra": "not allowed"},
        )


def test_prompt_contract_is_immutable_and_rejects_invalid_prompt_id():
    with pytest.raises(ValidationError):
        PromptTemplateContract(
            prompt_id="Agent Prompt With Spaces",
            version="1.0.0",
            template="hello",
        )

    with pytest.raises(ValidationError):
        BASE_AGENT_SYSTEM_PROMPT.version = "2.0.0"


def test_prompt_renderer_preserves_literal_braces_without_declared_variables():
    contract = PromptTemplateContract(
        prompt_id="test.literal-json",
        version="1.0.0",
        template='Return JSON like {"ok": true}.',
    )

    rendered = PromptRenderer().render(contract)
    assert rendered.content == 'Return JSON like {"ok": true}.'
