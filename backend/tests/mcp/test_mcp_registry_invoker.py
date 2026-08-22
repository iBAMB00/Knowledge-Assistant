import pytest

from app.agent.mcp.client import MCPRegistryInvoker
from app.agent.mcp.config import MCPServerConfig
from app.agent.mcp.contracts import MCPToolCallResult
from app.agent.mcp.registry import MCPServerRegistry


class FakeSession:
    async def list_tools(self):
        return [
            {
                "name": "echo",
                "description": "echo",
                "inputSchema": {"type": "object", "properties": {}},
            }
        ]

    async def call_tool(self, *, tool_name, arguments):
        return MCPToolCallResult(
            structured_content={"tool": tool_name, "arguments": arguments}
        )


class FakeConnectionManager:
    def __init__(self):
        self.disconnected = []

    async def connect(self, server_id):
        assert server_id == "demo"
        return FakeSession()

    async def disconnect(self, server_id):
        self.disconnected.append(server_id)


@pytest.mark.anyio
async def test_registry_invoker_discovers_stable_descriptors_and_disconnects():
    manager = FakeConnectionManager()
    invoker = MCPRegistryInvoker(connection_manager=manager)

    descriptors = await invoker.list_tools(server_id="demo")

    assert descriptors[0].exposed_name == "mcp__demo__echo"
    assert manager.disconnected == ["demo"]


def test_registry_invoker_executes_and_disconnects():
    manager = FakeConnectionManager()
    invoker = MCPRegistryInvoker(connection_manager=manager)

    result = invoker.call_tool(
        server_id="demo",
        tool_name="echo",
        arguments={"message": "hello"},
        context=None,
    )

    assert result.structured_content == {
        "tool": "echo",
        "arguments": {"message": "hello"},
    }
    assert manager.disconnected == ["demo"]
