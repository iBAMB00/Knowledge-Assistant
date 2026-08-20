"""STDIO MCP transport implementation boundary.

This file starts the concrete SDK integration layer.
The Agent Core continues to depend only on MCPTransportAdapter.
"""

from typing import Any

from app.agent.mcp.config import MCPServerConfig
from app.agent.mcp.contracts import MCPToolCallResult
from app.agent.mcp.transport import MCPClientSession, MCPTransportAdapter


class StdioMCPClientSession(MCPClientSession):
    """STDIO MCP session.

    The actual MCP SDK client/channel is intentionally isolated here.
    Future changes only replace internal connection handling.
    """

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self._initialized = False
        self._closed = False

    async def initialize(self) -> None:
        if self._closed:
            raise RuntimeError("MCP session already closed")

        # A2.2.1 keeps the lifecycle boundary stable.
        # Real SDK ClientSession initialization is wired in the next step.
        self._initialized = True

    async def list_tools(self) -> list[dict[str, Any]]:
        if not self._initialized:
            raise RuntimeError("MCP session is not initialized")

        return []

    async def call_tool(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> MCPToolCallResult:
        if not self._initialized:
            raise RuntimeError("MCP session is not initialized")

        return MCPToolCallResult(
            is_error=True,
            text_content=["MCP SDK ClientSession integration pending"],
        )

    async def close(self) -> None:
        self._closed = True


class StdioMCPTransportAdapter(MCPTransportAdapter):
    """Create sessions through STDIO transport."""

    def __init__(self, config: MCPServerConfig):
        self.config = config

    async def create_session(self) -> MCPClientSession:
        return StdioMCPClientSession(self.config)
