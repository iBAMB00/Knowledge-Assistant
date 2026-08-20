"""MCP Tool Invoker boundary tests."""

from unittest.mock import AsyncMock

import pytest

from app.agent.mcp.client import MCPClientInvoker
from app.agent.mcp.contracts import MCPToolCallResult
from app.agent.mcp.transport import MCPClientSessionManager


@pytest.mark.anyio
async def test_invoker_calls_mcp_session():
    session = AsyncMock()
    session.call_tool.return_value = MCPToolCallResult(
        structured_content={"result": "ok"}
    )

    class FakeManager:
        async def start(self):
            return session

    invoker = MCPClientInvoker(FakeManager())

    result = await invoker._call_tool_async(
        tool_name="echo_tool",
        arguments={"message": "hello"},
    )

    assert result.structured_content == {"result": "ok"}
    session.call_tool.assert_called_once()
