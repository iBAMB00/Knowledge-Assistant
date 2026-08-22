import pytest

from app.agent.mcp.runtime_initializer import MCPRuntimeInitializer
from app.agent.mcp.runtime_store import MCPRuntimeToolStore


@pytest.mark.anyio
async def test_runtime_initializer_publishes_loaded_tools():
    tool = type("FakeTool", (), {"name": "mcp__demo__echo"})()

    class FakeLoader:
        async def load_tools(self):
            return [tool]

    store = MCPRuntimeToolStore()
    initializer = MCPRuntimeInitializer(
        loader=FakeLoader(),
        tool_store=store,
    )

    result = await initializer.initialize()

    assert result == [tool]
    assert store.snapshot() == (tool,)

    initializer.clear()
    assert store.snapshot() == ()
