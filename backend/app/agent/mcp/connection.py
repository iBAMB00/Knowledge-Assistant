"""MCP connection lifecycle manager.

A3.2: manage MCP server connection lifecycle on top of server registry.
"""

from typing import Any

from app.agent.mcp.registry import MCPServerRegistry
from app.agent.mcp.transport import MCPClientSession, MCPClientSessionManager


class MCPConnectionManager:
    """Manage MCP sessions by server id.

    Registry owns server definitions.
    Connection manager owns active session lifecycle.
    """

    def __init__(
        self,
        *,
        registry: MCPServerRegistry,
        session_factory: Any,
    ) -> None:
        self._registry = registry
        self._session_factory = session_factory
        self._sessions: dict[str, MCPClientSessionManager] = {}

    async def connect(self, server_id: str) -> MCPClientSession:
        """Create or reuse a session for a registered MCP server."""
        config = self._registry.get(server_id)
        if config is None:
            raise ValueError(f"MCP server not found: {server_id}")

        manager = self._sessions.get(server_id)
        if manager is None:
            manager = self._session_factory(config)
            self._sessions[server_id] = manager

        return await manager.start()

    async def disconnect(self, server_id: str) -> None:
        """Close and remove a server connection."""
        manager = self._sessions.pop(server_id, None)
        if manager is not None:
            await manager.close()

    async def disconnect_all(self) -> None:
        """Close all active MCP server connections."""
        server_ids = list(self._sessions.keys())

        for server_id in server_ids:
            await self.disconnect(server_id)
