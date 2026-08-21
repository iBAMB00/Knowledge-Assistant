import pytest

from app.agent.mcp.loader import MCPToolLoader


@pytest.mark.anyio
async def test_loader_module_exists():
    assert MCPToolLoader is not None
