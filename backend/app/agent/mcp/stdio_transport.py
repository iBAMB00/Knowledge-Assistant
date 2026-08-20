"""STDIO MCP transport implementation.

Concrete MCP SDK details are isolated in this module. Agent Core should only
consume MCPClientSession abstraction.
"""

from typing import Any

from app.agent.mcp.config import MCPServerConfig
from app.agent.mcp.contracts import MCPToolCallResult
from app.agent.mcp.transport import MCPClientSession, MCPTransportAdapter


class StdioMCPClientSession(MCPClientSession):
    """MCP session backed by stdio transport.

    The actual SDK session object is injected after connection creation.
    This keeps lifecycle handling independent from Agent Core.
    """

    def __init__(self, sdk_session: Any, cleanup: Any):
        self._sdk_session = sdk_session
        self._cleanup = cleanup

    async def initialize(self) -> None:
        await self._sdk_session.initialize()

    async def list_tools(self) -> list[dict[str, Any]]:
        result = await self._sdk_session.list_tools()
        return [tool.model_dump() if hasattr(tool, "model_dump") else tool for tool in result.tools]

    async def call_tool(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> MCPToolCallResult:
        result = await self._sdk_session.call_tool(tool_name, arguments)
        return MCPToolCallResult.from_sdk_result(result)

    async def close(self) -> None:
        await self._cleanup()


class StdioMCPTransportAdapter(MCPTransportAdapter):
    """Create MCP sessions using stdio transport."""

    def __init__(self, config: MCPServerConfig):
        self.config = config

    async def create_session(self) -> MCPClientSession:
        # SDK transport creation is intentionally isolated here.
        # The concrete stdio_client wiring is completed when MCP SDK dependency
        # is finalized in the runtime layer.
        raise NotImplementedError(
            "Complete MCP SDK stdio_client wiring is required"
        )
