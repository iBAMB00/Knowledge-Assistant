"""MCP Tool Provider integration boundary for Agent Core."""

from app.agent.mcp.client import MCPToolInvoker
from app.agent.mcp.contracts import (
    MCPRemoteToolDescriptor,
    MCPToolAnnotations,
    MCPToolCallResult,
    build_mcp_exposed_tool_name,
)
from app.agent.mcp.tool import MCPBackedAgentTool, MCPRemoteToolError

__all__ = [
    "MCPBackedAgentTool",
    "MCPRemoteToolDescriptor",
    "MCPRemoteToolError",
    "MCPToolAnnotations",
    "MCPToolCallResult",
    "MCPToolInvoker",
    "build_mcp_exposed_tool_name",
    "MCPServerConfig",
    "MCPTransportType",
    "MCPClientSession",
    "MCPTransportAdapter",
    "MCPClientSessionManager",
]

from app.agent.mcp.config import MCPServerConfig, MCPTransportType
from app.agent.mcp.transport import (
    MCPClientSession,
    MCPTransportAdapter,
    MCPClientSessionManager,
)
