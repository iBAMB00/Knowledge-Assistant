"""A2.4 MCP tool discovery boundary tests."""

import pytest

from app.agent.mcp.client import MCPClientInvoker
from app.agent.mcp.contracts import MCPRemoteToolDescriptor, MCPToolCallResult


@pytest.mark.anyio
async def test_mcp_tool_discovery_converts_remote_schema():
    class FakeSession:
        async def list_tools(self):
            return [
                {
                    "name": "echo",
                    "description": "echo message",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "message": {"type": "string"}
                        },
                    },
                }
            ]

    class FakeManager:
        async def start(self):
            return FakeSession()

    invoker = MCPClientInvoker(FakeManager())

    tools = await invoker.list_tools(server_id="demo")

    assert len(tools) == 1
    assert isinstance(tools[0], MCPRemoteToolDescriptor)
    assert tools[0].remote_name == "echo"
