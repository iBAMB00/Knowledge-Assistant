"""MCP client invocation boundary.

A2.3/A2.4 bridge Agent Tool execution into MCP Client Session.
v2.2.2 adds a registry-aware invoker used by the application runtime.
"""

import asyncio
from typing import Any, Callable, Protocol

from app.agent.context import ToolExecutionContext
from app.agent.mcp.connection import MCPConnectionManager
from app.agent.mcp.contracts import MCPRemoteToolDescriptor, MCPToolCallResult
from app.agent.mcp.transport import MCPClientSessionManager


class MCPToolInvoker(Protocol):
    """MCP tool invocation contract used by Agent Tool layer."""

    def call_tool(
        self,
        *,
        server_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> MCPToolCallResult:
        ...

    async def list_tools(
        self,
        *,
        server_id: str,
    ) -> list[MCPRemoteToolDescriptor]:
        ...


class MCPClientInvoker:
    """Single-session-manager MCP invoker retained as the transport adapter baseline."""

    def __init__(self, session_manager: MCPClientSessionManager):
        self._session_manager = session_manager

    def call_tool(
        self,
        *,
        server_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> MCPToolCallResult:
        del server_id
        del context

        return asyncio.run(
            self._call_tool_async(
                tool_name=tool_name,
                arguments=arguments,
            )
        )

    async def _call_tool_async(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> MCPToolCallResult:
        session = await self._session_manager.start()

        return await session.call_tool(
            tool_name=tool_name,
            arguments=arguments,
        )

    async def list_tools(
        self,
        *,
        server_id: str,
    ) -> list[MCPRemoteToolDescriptor]:
        session = await self._session_manager.start()
        tools = await session.list_tools()
        return _build_remote_descriptors(server_id=server_id, tools=tools)


class MCPRegistryInvoker:
    """Route discovered MCP Tools to the server selected by stable ``server_id``.

    The current Agent Runtime is synchronous. Each MCP operation therefore owns
    one short-lived async session and closes it on the same event loop that
    created it. This avoids reusing SDK streams across FastAPI worker threads or
    across ``asyncio.run`` event loops. Persistent session pooling is deliberately
    deferred to the future Agent Runtime/Harness stage.
    """

    def __init__(self, *, connection_manager: MCPConnectionManager) -> None:
        self._connection_manager = connection_manager

    def call_tool(
        self,
        *,
        server_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> MCPToolCallResult:
        del context  # Trusted context is intentionally not forwarded as model arguments.

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError(
                "MCPRegistryInvoker.call_tool requires the synchronous Agent runtime"
            )

        return asyncio.run(
            self._call_tool_once(
                server_id=server_id,
                tool_name=tool_name,
                arguments=arguments,
            )
        )

    async def _call_tool_once(
        self,
        *,
        server_id: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> MCPToolCallResult:
        session = await self._connection_manager.connect(server_id)
        try:
            return await session.call_tool(
                tool_name=tool_name,
                arguments=arguments,
            )
        finally:
            await self._connection_manager.disconnect(server_id)

    async def list_tools(
        self,
        *,
        server_id: str,
    ) -> list[MCPRemoteToolDescriptor]:
        session = await self._connection_manager.connect(server_id)
        try:
            tools = await session.list_tools()
            return _build_remote_descriptors(
                server_id=server_id,
                tools=tools,
            )
        finally:
            await self._connection_manager.disconnect(server_id)


def _build_remote_descriptors(
    *,
    server_id: str,
    tools: list[dict[str, Any]],
) -> list[MCPRemoteToolDescriptor]:
    """Normalize SDK ``tools/list`` payloads into the stable host contract."""

    return [
        MCPRemoteToolDescriptor(
            server_id=server_id,
            remote_name=item["name"],
            title=item.get("title"),
            description=item.get("description"),
            input_schema=item.get("inputSchema", {"type": "object"}),
            output_schema=item.get("outputSchema"),
            annotations=item.get("annotations") or {},
        )
        for item in tools
    ]
