import pytest

from app.agent.mcp.config import MCPServerConfig
from app.agent.mcp.registry import MCPServerRegistry
from app.agent.mcp.connection import MCPConnectionManager


class FakeSession:
    def __init__(self):
        self.initialized = False
        self.closed = False

    async def initialize(self):
        self.initialized = True

    async def close(self):
        self.closed = True


class FakeSessionManager:
    def __init__(self, config):
        self.config = config
        self.session = FakeSession()

    async def start(self):
        if not self.session.initialized:
            await self.session.initialize()
        return self.session

    async def close(self):
        await self.session.close()


@pytest.mark.asyncio
async def test_connection_manager_reuses_session():
    registry = MCPServerRegistry()
    registry.register(
        MCPServerConfig(
            server_id="demo",
            command="python",
        )
    )

    created = []

    def factory(config):
        manager = FakeSessionManager(config)
        created.append(manager)
        return manager

    manager = MCPConnectionManager(
        registry=registry,
        session_factory=factory,
    )

    first = await manager.connect("demo")
    second = await manager.connect("demo")

    assert first is second
    assert len(created) == 1


@pytest.mark.asyncio
async def test_connection_manager_disconnects():
    registry = MCPServerRegistry()
    registry.register(
        MCPServerConfig(
            server_id="demo",
            command="python",
        )
    )

    created = []

    def factory(config):
        manager = FakeSessionManager(config)
        created.append(manager)
        return manager

    connection_manager = MCPConnectionManager(
        registry=registry,
        session_factory=factory,
    )

    await connection_manager.connect("demo")
    await connection_manager.disconnect("demo")

    assert created[0].session.closed is True
