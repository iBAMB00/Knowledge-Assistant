"""MCP dynamic tool loader.

A3.4: compose registry, connection, discovery and runtime builder.
"""

from app.agent.mcp.registry import MCPServerRegistry
from app.agent.mcp.connection import MCPConnectionManager
from app.agent.mcp.namespace import MCPToolNamespaceRegistry
from app.agent.mcp.runtime import MCPToolRuntimeBuilder


class MCPToolLoader:
    """Load MCP tools into Agent runtime."""

    def __init__(
        self,
        *,
        registry: MCPServerRegistry,
        connection_manager: MCPConnectionManager,
        namespace_registry: MCPToolNamespaceRegistry,
        runtime_builder: MCPToolRuntimeBuilder,
    ) -> None:
        self._registry = registry
        self._connection_manager = connection_manager
        self._namespace_registry = namespace_registry
        self._runtime_builder = runtime_builder

    async def load_tools(self):
        descriptors = []

        for server in self._registry.list_servers():
            client = await self._connection_manager.connect(
                server.server_id
            )

            tools = await client.list_tools()

            for tool in tools:
                self._namespace_registry.register_tool(
                    server_id=server.server_id,
                    tool_name=tool.remote_name,
                )

            descriptors.extend(tools)

        return self._runtime_builder.build_runtime_tools(descriptors)
