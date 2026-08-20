"""STDIO MCP transport implementation."""

from contextlib import AsyncExitStack
from typing import Any

from app.agent.mcp.config import MCPServerConfig
from app.agent.mcp.contracts import MCPToolCallResult
from app.agent.mcp.transport import MCPClientSession, MCPTransportAdapter


class StdioMCPClientSession(MCPClientSession):
    """MCP session backed by the official SDK client session."""

    def __init__(self, sdk_session: Any, exit_stack: AsyncExitStack):
        self._sdk_session = sdk_session
        self._exit_stack = exit_stack

    async def initialize(self) -> None:
        await self._sdk_session.initialize()

    async def list_tools(self) -> list[dict[str, Any]]:
        result = await self._sdk_session.list_tools()
        return [
            item.model_dump() if hasattr(item, "model_dump") else item
            for item in result.tools
        ]

    async def call_tool(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> MCPToolCallResult:
        result = await self._sdk_session.call_tool(tool_name, arguments)
        return MCPToolCallResult.from_sdk_result(result)

    async def close(self) -> None:
        await self._exit_stack.aclose()


class StdioMCPTransportAdapter(MCPTransportAdapter):
    """Create MCP sessions through stdio transport."""

    def __init__(self, config: MCPServerConfig):
        self.config = config

    async def create_session(self) -> MCPClientSession:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        exit_stack = AsyncExitStack()

        server_params = StdioServerParameters(
            command=self.config.command,
            args=self.config.args,
        )

        read_stream, write_stream = await exit_stack.enter_async_context(
            stdio_client(server_params)
        )

        sdk_session = await exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )

        return StdioMCPClientSession(
            sdk_session=sdk_session,
            exit_stack=exit_stack,
        )
