"""A3.1 MCP server registry tests."""

import pytest

from app.agent.mcp.config import MCPServerConfig
from app.agent.mcp.registry import MCPServerRegistry


def test_register_and_get_server():
    registry = MCPServerRegistry()
    config = MCPServerConfig(
        server_id="knowledge",
        command="python",
    )

    registry.register(config)

    assert registry.get("knowledge") == config


def test_duplicate_server_id_is_rejected():
    registry = MCPServerRegistry()
    config = MCPServerConfig(server_id="knowledge")

    registry.register(config)

    with pytest.raises(ValueError):
        registry.register(config)


def test_remove_server():
    registry = MCPServerRegistry()
    config = MCPServerConfig(server_id="knowledge")

    registry.register(config)

    removed = registry.remove("knowledge")

    assert removed == config
    assert registry.get("knowledge") is None
