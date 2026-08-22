import pytest

from app.agent.mcp.contracts import MCPRemoteToolDescriptor
from app.agent.mcp.loader import MCPToolLoader
from app.agent.mcp.namespace import MCPToolNamespaceRegistry
from app.agent.mcp.registry import MCPServerRegistry
from app.agent.mcp.config import MCPServerConfig


@pytest.mark.anyio
async def test_loader_discovers_and_builds_namespaced_runtime_tool():
    registry = MCPServerRegistry()
    registry.register(
        MCPServerConfig(
            server_id="demo",
            command="python",
        )
    )

    descriptor = MCPRemoteToolDescriptor(
        server_id="demo",
        remote_name="echo",
        description="echo",
        input_schema={"type": "object", "properties": {}},
    )

    class FakeInvoker:
        async def list_tools(self, *, server_id):
            assert server_id == "demo"
            return [descriptor]

    class FakeRuntimeBuilder:
        def build_runtime_tools(self, descriptors):
            assert descriptors == [descriptor]
            return [type("FakeTool", (), {"name": descriptor.exposed_name})()]

    namespace_registry = MCPToolNamespaceRegistry()
    loader = MCPToolLoader(
        registry=registry,
        invoker=FakeInvoker(),
        namespace_registry=namespace_registry,
        runtime_builder=FakeRuntimeBuilder(),
    )

    tools = await loader.load_tools()

    assert tools[0].name == "mcp__demo__echo"
    assert namespace_registry.resolve("mcp__demo__echo") == ("demo", "echo")
