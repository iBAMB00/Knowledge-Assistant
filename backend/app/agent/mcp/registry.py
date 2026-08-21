"""MCP server registry.

Responsible for storing MCP server definitions and providing lookup capability.
"""

from app.agent.mcp.config import MCPServerConfig


class MCPServerRegistry:
    """In-memory registry for MCP server configurations."""

    def __init__(self) -> None:
        self._servers: dict[str, MCPServerConfig] = {}

    def register(self, config: MCPServerConfig) -> None:
        """Register a server configuration.

        Existing server ids are not silently overwritten to avoid runtime
        configuration changes without explicit action.
        """
        if config.server_id in self._servers:
            raise ValueError(f"MCP server already exists: {config.server_id}")

        self._servers[config.server_id] = config

    def get(self, server_id: str) -> MCPServerConfig | None:
        return self._servers.get(server_id)

    def list_servers(self) -> list[MCPServerConfig]:
        return list(self._servers.values())

    def remove(self, server_id: str) -> MCPServerConfig | None:
        return self._servers.pop(server_id, None)
