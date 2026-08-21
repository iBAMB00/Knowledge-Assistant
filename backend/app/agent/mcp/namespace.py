"""MCP tool namespace registry.

Provides stable names for MCP tools inside Agent runtime.
"""


class MCPToolNamespaceRegistry:
    """Registry for mapping runtime tool names to MCP remote tools."""

    def __init__(self) -> None:
        self._tools: dict[str, tuple[str, str]] = {}

    def register_tool(
        self,
        *,
        server_id: str,
        tool_name: str,
    ) -> str:
        runtime_name = self._build_name(
            server_id=server_id,
            tool_name=tool_name,
        )

        if runtime_name in self._tools:
            raise ValueError(
                f"MCP tool namespace already exists: {runtime_name}"
            )

        self._tools[runtime_name] = (server_id, tool_name)
        return runtime_name

    def resolve(
        self,
        runtime_name: str,
    ) -> tuple[str, str] | None:
        return self._tools.get(runtime_name)

    def list_tools(self) -> dict[str, tuple[str, str]]:
        return dict(self._tools)

    @staticmethod
    def _build_name(
        *,
        server_id: str,
        tool_name: str,
    ) -> str:
        return f"mcp__{server_id}__{tool_name}"
