"""MCP connection configuration boundary."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class MCPTransportType(str, Enum):
    STDIO = "stdio"


class MCPServerConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    server_id: str = Field(min_length=1, max_length=32)
    transport: MCPTransportType = MCPTransportType.STDIO
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    timeout_seconds: float = Field(default=30, gt=0, le=300)
