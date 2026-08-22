"""v2.2 Release Gate 专用真实 MCP stdio Server。"""

from pydantic import BaseModel

from mcp.server.fastmcp import FastMCP


mcp = FastMCP("Knowledge Assistant MCP Release Probe")


class EchoResult(BaseModel):
    result: str


@mcp.tool()
def echo(message: str) -> EchoResult:
    """Echo the exact message for the Knowledge Assistant MCP release probe."""

    return EchoResult(result=message)


if __name__ == "__main__":
    mcp.run()
