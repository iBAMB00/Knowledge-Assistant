"""A2.4 discovery to Agent Tool adapter tests."""
from app.agent.mcp.discovery import MCPToolDiscoveryService
from app.agent.mcp.contracts import MCPRemoteToolDescriptor


def test_discovery_builds_agent_tool():
    descriptor = MCPRemoteToolDescriptor(
        server_id="demo",
        remote_name="echo",
        description="echo message",
        input_schema={
            "type": "object",
            "properties": {
                "message": {"type": "string"}
            },
        },
    )

    service = MCPToolDiscoveryService(invoker=object())
    tools = service.build_tools([descriptor])

    assert len(tools) == 1
    assert tools[0].name == descriptor.exposed_name
