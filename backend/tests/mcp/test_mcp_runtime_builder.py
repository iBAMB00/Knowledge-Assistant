"""A2.4 MCP runtime builder tests."""

from app.agent.mcp.discovery import MCPToolDiscoveryService
from app.agent.mcp.runtime import MCPToolRuntimeBuilder
from app.agent.mcp.contracts import MCPRemoteToolDescriptor


def test_mcp_runtime_builder_creates_agent_tools():
    class FakeInvoker:
        pass

    service = MCPToolDiscoveryService(
        invoker=FakeInvoker()
    )

    builder = MCPToolRuntimeBuilder(service)

    tools = builder.build_runtime_tools(
        [
            MCPRemoteToolDescriptor(
                server_id="demo",
                remote_name="echo",
                description="echo",
                input_schema={"type": "object"},
            )
        ]
    )

    assert len(tools) == 1
    assert tools[0].name.endswith("__echo")
