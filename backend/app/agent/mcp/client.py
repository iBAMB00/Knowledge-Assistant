from typing import Any, Protocol

from app.agent.context import ToolExecutionContext
from app.agent.mcp.contracts import MCPToolCallResult


class MCPToolInvoker(Protocol):
    """
    Agent Core 使用的 MCP 调用边界。

    A1 只定义同步、framework-neutral contract；A2 的真实 MCP SDK / transport
    需要在该边界后完成 session、async bridge、认证和结果归一化。

    Trusted Context 必须作为独立参数传入，不能混进模型生成的 arguments。
    """

    def call_tool(
        self,
        *,
        server_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> MCPToolCallResult:
        """执行一次已经通过本地主机 Schema 校验的远端 MCP Tool。"""
