import pytest

from app.agent.mcp.health import MCPHealthMonitor


def test_health_monitor_status():
    monitor = MCPHealthMonitor()

    monitor.set_status("demo", "READY")

    assert monitor.is_healthy("demo")


class FakeInitializer:
    def __init__(self, events=None):
        self.events = events

    async def initialize(self):
        if self.events is not None:
            self.events.append("initialize")
        return []

    def clear(self):
        if self.events is not None:
            self.events.append("clear")


class FakeConnectionManager:
    def __init__(self):
        self.closed = False

    async def disconnect_all(self):
        self.closed = True


class FakeRegistryBootstrapper:
    def __init__(self, events):
        self.events = events

    def restore(self):
        self.events.append("restore")


@pytest.mark.anyio
async def test_lifecycle_restores_registry_before_runtime_initialization():
    from app.agent.mcp.lifecycle import MCPLifecycleManager

    events = []
    manager = MCPLifecycleManager(
        runtime_initializer=FakeInitializer(events),
        connection_manager=FakeConnectionManager(),
        registry_bootstrapper=FakeRegistryBootstrapper(events),
    )

    await manager.startup()

    assert events == ["restore", "initialize"]


@pytest.mark.anyio
async def test_lifecycle_shutdown():
    from app.agent.mcp.lifecycle import MCPLifecycleManager

    manager = MCPLifecycleManager(
        runtime_initializer=FakeInitializer(),
        connection_manager=FakeConnectionManager(),
    )

    await manager.shutdown()

    assert manager._connection_manager.closed
