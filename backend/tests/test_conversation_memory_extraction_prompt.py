from app.agent.agent_prompt import render_conversation_memory_extraction_prompt


def test_memory_extraction_prompt_has_stable_contract_and_safety_boundary() -> None:
    prompt = render_conversation_memory_extraction_prompt()

    assert prompt.prompt_id == "agent.conversation-memory-extraction"
    assert prompt.version == "1.0.0"
    assert "用户明确表达" in prompt.content
    assert "企业文档" in prompt.content
    assert "Tool Result" in prompt.content
    assert "API Key" in prompt.content
    assert "只输出 JSON" in prompt.content
