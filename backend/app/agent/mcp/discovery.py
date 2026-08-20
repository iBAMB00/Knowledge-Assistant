"""MCP tool discovery to Agent Tool adapter boundary.

A2.4: convert discovered MCP descriptors into runtime Agent Tools.
"""
from app.agent.mcp.contracts import MCPRemoteToolDescriptor
from app.agent.mcp.tool import MCPBackedAgentTool
from app.agent.tools.base import ToolRiskLevel


class MCPToolDiscoveryService:
    """Build Agent Tools from MCP discovery results.

    Discovery only creates local adapters. It does not decide whether
    a remote tool is trusted; risk approval remains explicit.
    """

    def __init__(self, *, invoker):
        self._invoker = invoker

    def build_tools(
        self,
        descriptors: list[MCPRemoteToolDescriptor],
    ) -> list[MCPBackedAgentTool]:
        return [
            MCPBackedAgentTool(
                descriptor=descriptor,
                invoker=self._invoker,
                approved_risk_level=ToolRiskLevel.READ_ONLY,
            )
            for descriptor in descriptors
        ]
