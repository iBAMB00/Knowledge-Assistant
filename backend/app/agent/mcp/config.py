"""MCP connection configuration boundary.

A2.1 only defines the connection boundary. Real SDK/transport implementations
are added behind this layer in later A2 steps.
"""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class MCPTransportType(str, Enum):
    """Supported MCP transport names for the first implementation stage."""

    STDIO = "stdio"


class MCPServerConfig(BaseModel):
    """Configuration of one external MCP server.

    v2.2-A2.1 intentionally keeps the scope small:
    - one server
    - one transport
    - no dynamic registry
    - no OAuth lifecycle
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    server_id: str = Field(min_length=1, max_length=32)
    transport: MCPTransportType = Field(default=MCPTransportType.STDIO)
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    timeout_seconds: float = Field(default=30, gt=0, le=300)
