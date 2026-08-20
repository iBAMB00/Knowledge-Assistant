"""MCP transport abstraction.

Agent Core should not depend on a concrete MCP SDK transport. Concrete
ClientSession implementations are introduced behind this adapter.
"""

from abc import ABC, abstractmethod
from typing import Any

from app.agent.mcp.contracts import MCPToolCallResult


class MCPClientSession(ABC):
    """Framework-neutral MCP session lifecycle."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize MCP protocol session."""

    @abstractmethod
    async def list_tools(self) -> list[dict[str, Any]]:
        """Fetch tools/list result from MCP server."""

    @abstractmethod
    async def call_tool(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> MCPToolCallResult:
        """Execute tools/call and normalize response."""

    @abstractmethod
    async def close(self) -> None:
        """Release MCP resources."""


class MCPTransportAdapter(ABC):
    """Create MCP sessions without leaking transport details."""

    @abstractmethod
    async def create_session(self) -> MCPClientSession:
        """Create a protocol session."""


class MCPClientSessionManager:
    """Manage MCP session lifecycle for future concrete transports.

    A2.1 only provides lifecycle ownership. Concrete SDK connection logic
    remains inside MCPTransportAdapter implementations.
    """

    def __init__(self, transport: MCPTransportAdapter):
        self._transport = transport
        self._session: MCPClientSession | None = None

    async def start(self) -> MCPClientSession:
        if self._session is None:
            self._session = await self._transport.create_session()
            await self._session.initialize()
        return self._session

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None
