"""MCP client invocation boundary.

A2.3/A2.4: bridge Agent Tool execution into MCP Client Session.
"""

import asyncio
from typing import Any

from app.agent.context import ToolExecutionContext
from app.agent.mcp.contracts import MCPToolCallResult
from app.agent.mcp.transport import MCPClientSessionManager


class MCPClientInvoker:
    """Concrete MCPToolInvoker implementation.

    Agent Core remains synchronous because BaseAgentTool.execute is sync.
    The MCP SDK runtime is async, therefore this class owns the async bridge.
    """

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
