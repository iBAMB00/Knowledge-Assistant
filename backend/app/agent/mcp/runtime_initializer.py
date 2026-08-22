"""MCP runtime integration entry.

A3.5 provides one initialization boundary for loading MCP tools into the
application-level Agent Tool snapshot.
"""

from app.agent.mcp.loader import MCPToolLoader
from app.agent.mcp.runtime_store import MCPRuntimeToolStore
from app.agent.tools.base import BaseAgentTool


class MCPRuntimeInitializer:
    """Load MCP tools at startup and publish an immutable runtime snapshot."""

    def __init__(
        self,
        *,
        loader: MCPToolLoader,
        tool_store: MCPRuntimeToolStore | None = None,
    ) -> None:
        self._loader = loader
        self._tool_store = tool_store

    async def initialize(self) -> list[BaseAgentTool]:
        tools = await self._loader.load_tools()
        if self._tool_store is not None:
            self._tool_store.replace(tools)
        return tools

    def clear(self) -> None:
        if self._tool_store is not None:
            self._tool_store.clear()
