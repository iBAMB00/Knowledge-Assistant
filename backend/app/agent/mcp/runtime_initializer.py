"""MCP runtime integration entry.

A3.5: provide a single initialization boundary for loading MCP tools
into Agent runtime.
"""

from app.agent.mcp.loader import MCPToolLoader
from app.agent.tools.base import BaseAgentTool


class MCPRuntimeInitializer:
    """Initialize MCP tools and expose runtime-ready tools."""

    def __init__(
        self,
        *,
        loader: MCPToolLoader,
    ) -> None:
        self._loader = loader

    async def initialize(self) -> list[BaseAgentTool]:
        """Load MCP tools during agent runtime startup."""
        return await self._loader.load_tools()
