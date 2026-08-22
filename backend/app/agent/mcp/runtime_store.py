"""Application-level MCP runtime Tool snapshot.

v2.2.2 keeps the loaded MCP Toolset immutable during one application
lifecycle. Dynamic refresh remains a deferred management-platform capability.
"""

from collections.abc import Sequence

from app.agent.tools.base import BaseAgentTool


class MCPRuntimeToolStore:
    """Store the MCP tools discovered during application startup."""

    def __init__(self) -> None:
        self._tools: tuple[BaseAgentTool, ...] = ()

    def replace(self, tools: Sequence[BaseAgentTool]) -> None:
        names = [tool.name for tool in tools]
        if len(names) != len(set(names)):
            raise ValueError("duplicate MCP runtime tool name")
        self._tools = tuple(tools)

    def snapshot(self) -> tuple[BaseAgentTool, ...]:
        return self._tools

    def clear(self) -> None:
        self._tools = ()
