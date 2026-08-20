"""MCP Client Runtime validation tests."""

import anyio
import pytest


@pytest.mark.anyio
async def test_mcp_runtime_contract_placeholder():
    assert True


@pytest.mark.anyio
async def test_fake_mcp_echo_tool_contract():
    from tests.mcp.fake_mcp_server import echo_tool

    result = await echo_tool({"message": "hello"})

    assert result == {"result": "hello"}


@pytest.mark.anyio
async def test_fake_mcp_timeout_case():
    from tests.mcp.fake_mcp_server import slow_tool

    with pytest.raises(TimeoutError):
        with anyio.fail_after(0.01):
            await slow_tool({"delay": 0.1})


@pytest.mark.anyio
async def test_session_requires_initialize_before_use():
    """
    A2 lifecycle rule:
    tools/list and tools/call cannot bypass initialize handshake.
    """
    from unittest.mock import AsyncMock

    from app.agent.mcp.stdio_transport import StdioMCPClientSession

    session = StdioMCPClientSession(
        sdk_session=AsyncMock(),
        exit_stack=anyio.create_task_group,
    )

    with pytest.raises(RuntimeError):
        await session.list_tools()
