from app.api.dependencies.agent import _merge_agent_tools


class FakeTool:
    def __init__(self, name: str):
        self.name = name


def test_agent_tool_merge_keeps_local_and_mcp_tools():
    local = FakeTool("search_knowledge")
    mcp = FakeTool("mcp__demo__echo")

    result = _merge_agent_tools((local,), (mcp,))

    assert [tool.name for tool in result] == [
        "search_knowledge",
        "mcp__demo__echo",
    ]


def test_agent_tool_merge_rejects_name_collision():
    first = FakeTool("same")
    second = FakeTool("same")

    try:
        _merge_agent_tools((first,), (second,))
    except RuntimeError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("Agent Tool name collision must fail")
