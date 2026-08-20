"""STDIO MCP transport implementation boundary.

This module intentionally isolates concrete transport details from Agent Core.
"""

from app.agent.mcp.config import MCPServerConfig
from app.agent.mcp.transport import MCPClientSession, MCPTransportAdapter


class StdioMCPTransportAdapter(MCPTransportAdapter):
    """Create MCP sessions through stdio transport.

    The concrete MCP SDK wiring is kept behind this adapter so Agent Core,
    ToolDispatcher and MCPBackedAgentTool do not depend on SDK details.
    """

    def __init__(self, config: MCPServerConfig):
        self.config = config

    async def create_session(self) -> MCPClientSession:
        raise NotImplementedError(
            "STDIO MCP SDK session wiring is implemented in the next A2 step"
        )
