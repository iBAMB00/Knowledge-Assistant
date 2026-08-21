"""A2.4 MCP runtime execution integration tests."""

from app.agent.mcp.contracts import MCPRemoteToolDescriptor, MCPToolCallResult
from app.agent.mcp.discovery import MCPToolDiscoveryService
from app.agent.mcp.runtime import MCPToolRuntimeBuilder
from app.agent.context import ToolExecutionContext


def test_mcp_runtime_tool_executes_through_invoker():
    calls = []

    class FakeInvoker:
        def call_tool(self, *, server_id, tool_name, arguments, context):
            calls.append(
                {
                    "server_id": server_id,
                    "tool_name": tool_name,
                    "arguments": arguments,
                }
            )
            return MCPToolCallResult(
                structured_content={"message": "ok"},
                text_content=["ok"],
                is_error=False,
            )

    descriptor = MCPRemoteToolDescriptor(
        server_id="demo",
        remote_name="echo",
        description="echo",
        input_schema={
            "type": "object",
            "properties": {
                "message": {"type": "string"}
            },
        },
    )

    service = MCPToolDiscoveryService(invoker=FakeInvoker())
    builder = MCPToolRuntimeBuilder(service)

    tool = builder.build_runtime_tools([descriptor])[0]

    arguments = tool.validate_input({"message": "hello"})

    result = tool.execute(
        db=None,
        context=ToolExecutionContext(
            user_id=1,
            knowledge_base_id=1,
            role="admin",
            request_id="test-request-id",
        ),
        tool_input=arguments,
    )

    assert result.text_content == ["ok"]
    assert calls[0]["tool_name"] == "echo"
    assert calls[0]["arguments"]["message"] == "hello"
