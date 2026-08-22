from app.agent.mcp.runtime_store import MCPRuntimeToolStore


class FakeTool:
    def __init__(self, name: str):
        self.name = name


def test_runtime_tool_store_replaces_and_clears_snapshot():
    store = MCPRuntimeToolStore()
    first = FakeTool("mcp__demo__echo")

    store.replace([first])
    assert store.snapshot() == (first,)

    store.clear()
    assert store.snapshot() == ()


def test_runtime_tool_store_rejects_duplicate_names():
    store = MCPRuntimeToolStore()

    try:
        store.replace([FakeTool("same"), FakeTool("same")])
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate MCP runtime tool names must fail")
