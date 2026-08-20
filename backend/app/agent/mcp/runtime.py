"""MCP discovery runtime helpers.

A2.4: bridge discovered MCP descriptors into Agent Tool runtime inputs.
"""

from collections.abc import Sequence

from app.agent.mcp.contracts import MCPRemoteToolDescriptor
from app.agent.mcp.discovery import MCPToolDiscoveryService
from app.agent.tools.base import BaseAgentTool


class MCPToolRuntimeBuilder:
    """Build runtime-ready Agent Tools from MCP discovery results.

    This class intentionally does not own registration or dispatch.
    ToolDispatcher remains responsible for execution.
    """

    def __init__(self, discovery_service: MCPToolDiscoveryService):
        self._discovery_service = discovery_service

    def build_runtime_tools(
        self,
        descriptors: Sequence[MCPRemoteToolDescriptor],
    ) -> list[BaseAgentTool]:
        return self._discovery_service.build_tools(
            list(descriptors)
        )
