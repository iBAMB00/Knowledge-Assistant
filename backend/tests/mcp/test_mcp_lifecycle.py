import pytest

from app.agent.mcp.health import MCPHealthMonitor


def test_health_monitor_status():
    monitor = MCPHealthMonitor()

    monitor.set_status("demo", "READY")

    assert monitor.is_healthy("demo")


class FakeInitializer:
    async def initialize(self):
        return []


class FakeConnectionManager:
    def __init__(self):
        self.closed = False

    async def disconnect_all(self):
        self.closed = True


@pytest.mark.anyio
async def test_lifecycle_shutdown():
    from app.agent.mcp.lifecycle import MCPLifecycleManager

    manager = MCPLifecycleManager(
        runtime_initializer=FakeInitializer(),
        connection_manager=FakeConnectionManager(),
    )

    await manager.shutdown()

    assert manager._connection_manager.closed
