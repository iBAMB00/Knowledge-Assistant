"""STDIO MCP transport adapter.

A2.2 starts the concrete transport implementation behind the existing
MCPTransportAdapter boundary. The SDK/session details are isolated here.
"""

import asyncio
from typing import Any

from app.agent.mcp.config import MCPServerConfig
from app.agent.mcp.contracts import MCPToolCallResult
from app.agent.mcp.transport import MCPClientSession, MCPTransportAdapter


class StdioMCPClientSession(MCPClientSession):
    """Minimal stdio session placeholder for MCP SDK integration.

    The session lifecycle is implemented here so later MCP SDK wiring only
    replaces the internal process/channel handling.
    """

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self._initialized = False
        self._closed = False

    async def initialize(self) -> None:
        if self._closed:
            raise RuntimeError("MCP session already closed")
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
            success=False,
            error_type="transport_not_connected",
            message="MCP SDK call implementation is pending",
        )

    async def close(self) -> None:
        self._closed = True


class StdioMCPTransportAdapter(MCPTransportAdapter):
    """Create MCP sessions using stdio transport configuration."""

    def __init__(self, config: MCPServerConfig):
        self.config = config

    async def create_session(self) -> MCPClientSession:
        return StdioMCPClientSession(self.config)
