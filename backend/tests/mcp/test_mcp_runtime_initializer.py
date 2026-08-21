import pytest

from app.agent.mcp.runtime_initializer import MCPRuntimeInitializer


@pytest.mark.anyio
async def test_runtime_initializer_calls_loader():
    class FakeLoader:
        async def load_tools(self):
            return []

    initializer = MCPRuntimeInitializer(loader=FakeLoader())

    result = await initializer.initialize()

    assert result == []
