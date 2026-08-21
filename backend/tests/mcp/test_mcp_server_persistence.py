import pytest

from app.agent.mcp.config import MCPServerConfig
from app.agent.mcp.registry import MCPServerRegistry
from app.repositories.mcp_server_repository import MCPServerRepository
from app.services.mcp_server_persistence_service import (
    MCPRegistryRestoreConflictError,
    MCPServerPersistenceService,
)


def build_service(registry: MCPServerRegistry) -> MCPServerPersistenceService:
    return MCPServerPersistenceService(
        repository=MCPServerRepository(),
        registry=registry,
    )


def test_persisted_mcp_server_can_restore_registry(db):
    registry = MCPServerRegistry()
    service = build_service(registry)
    config = MCPServerConfig(
        server_id="demo",
        command="python",
        args=["server.py"],
        timeout_seconds=15,
    )

    service.create_server(db, config=config)
    restored = service.restore_registry(db)

    assert restored == [config]
    assert registry.get("demo") == config


def test_disabled_mcp_server_is_not_restored(db):
    registry = MCPServerRegistry()
    service = build_service(registry)
    config = MCPServerConfig(server_id="disabled", command="python")

    service.create_server(db, config=config, enabled=False)
    service.restore_registry(db)

    assert registry.get("disabled") is None


def test_restore_is_idempotent_for_same_config(db):
    registry = MCPServerRegistry()
    service = build_service(registry)
    config = MCPServerConfig(server_id="demo", command="python")

    service.create_server(db, config=config)
    service.restore_registry(db)
    restored_again = service.restore_registry(db)

    assert restored_again == []
    assert registry.list_servers() == [config]


def test_restore_fails_closed_on_runtime_config_conflict(db):
    registry = MCPServerRegistry()
    service = build_service(registry)
    persisted = MCPServerConfig(server_id="demo", command="python")

    service.create_server(db, config=persisted)
    registry.register(MCPServerConfig(server_id="demo", command="python3"))

    with pytest.raises(MCPRegistryRestoreConflictError):
        service.restore_registry(db)
