from app.agent.mcp.namespace import MCPToolNamespaceRegistry


def test_namespace_registry_creates_unique_runtime_name():
    registry = MCPToolNamespaceRegistry()

    name = registry.register_tool(
        server_id="demo",
        tool_name="echo",
    )

    assert name == "mcp__demo__echo"
    assert registry.resolve(name) == ("demo", "echo")


def test_namespace_registry_prevents_duplicate():
    registry = MCPToolNamespaceRegistry()

    registry.register_tool(
        server_id="demo",
        tool_name="echo",
    )

    try:
        registry.register_tool(
            server_id="demo",
            tool_name="echo",
        )
        assert False
    except ValueError:
        assert True
