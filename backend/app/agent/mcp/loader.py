"""MCP dynamic tool loader.

A3.4 composes persistent server definitions, discovery, namespace validation
and runtime Tool adaptation.
"""

from app.agent.mcp.client import MCPToolInvoker
from app.agent.mcp.namespace import MCPToolNamespaceRegistry
from app.agent.mcp.registry import MCPServerRegistry
from app.agent.mcp.runtime import MCPToolRuntimeBuilder


class MCPToolLoader:
    """Discover enabled MCP server tools and build Agent runtime adapters."""

    def __init__(
        self,
        *,
        registry: MCPServerRegistry,
        invoker: MCPToolInvoker,
        namespace_registry: MCPToolNamespaceRegistry,
        runtime_builder: MCPToolRuntimeBuilder,
    ) -> None:
        self._registry = registry
        self._invoker = invoker
        self._namespace_registry = namespace_registry
        self._runtime_builder = runtime_builder

    async def load_tools(self):
        descriptors = []

        for server in self._registry.list_servers():
            tools = await self._invoker.list_tools(
                server_id=server.server_id,
            )

            for tool in tools:
                runtime_name = self._namespace_registry.register_tool(
                    server_id=server.server_id,
                    tool_name=tool.remote_name,
                )
                if runtime_name != tool.exposed_name:
                    raise RuntimeError(
                        "MCP namespace contract mismatch: "
                        f"{server.server_id}/{tool.remote_name}"
                    )

            descriptors.extend(tools)

        return self._runtime_builder.build_runtime_tools(descriptors)
