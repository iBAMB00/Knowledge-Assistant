"""Application composition root for MCP runtime dependencies."""

from functools import lru_cache

from app.agent.mcp.client import MCPRegistryInvoker
from app.agent.mcp.connection import MCPConnectionManager
from app.agent.mcp.discovery import MCPToolDiscoveryService
from app.agent.mcp.lifecycle import MCPLifecycleManager
from app.agent.mcp.loader import MCPToolLoader
from app.agent.mcp.namespace import MCPToolNamespaceRegistry
from app.agent.mcp.registry import MCPServerRegistry
from app.agent.mcp.runtime import MCPToolRuntimeBuilder
from app.agent.mcp.runtime_initializer import MCPRuntimeInitializer
from app.agent.mcp.runtime_store import MCPRuntimeToolStore
from app.agent.mcp.stdio_transport import StdioMCPTransportAdapter
from app.agent.mcp.transport import MCPClientSessionManager
from app.core.database import SessionLocal
from app.repositories.mcp_server_repository import MCPServerRepository
from app.services.mcp_registry_bootstrap_service import MCPRegistryBootstrapService
from app.services.mcp_server_persistence_service import MCPServerPersistenceService


@lru_cache
def get_mcp_server_registry() -> MCPServerRegistry:
    return MCPServerRegistry()


@lru_cache
def get_mcp_namespace_registry() -> MCPToolNamespaceRegistry:
    return MCPToolNamespaceRegistry()


@lru_cache
def get_mcp_runtime_tool_store() -> MCPRuntimeToolStore:
    return MCPRuntimeToolStore()


def _build_session_manager(config) -> MCPClientSessionManager:
    return MCPClientSessionManager(
        StdioMCPTransportAdapter(config)
    )


@lru_cache
def get_mcp_connection_manager() -> MCPConnectionManager:
    return MCPConnectionManager(
        registry=get_mcp_server_registry(),
        session_factory=_build_session_manager,
    )


@lru_cache
def get_mcp_runtime_invoker() -> MCPRegistryInvoker:
    return MCPRegistryInvoker(
        connection_manager=get_mcp_connection_manager(),
    )


@lru_cache
def get_mcp_tool_loader() -> MCPToolLoader:
    invoker = get_mcp_runtime_invoker()
    return MCPToolLoader(
        registry=get_mcp_server_registry(),
        invoker=invoker,
        namespace_registry=get_mcp_namespace_registry(),
        runtime_builder=MCPToolRuntimeBuilder(
            MCPToolDiscoveryService(invoker=invoker)
        ),
    )


@lru_cache
def get_mcp_lifecycle_manager() -> MCPLifecycleManager:
    registry = get_mcp_server_registry()
    persistence_service = MCPServerPersistenceService(
        repository=MCPServerRepository(),
        registry=registry,
    )
    bootstrapper = MCPRegistryBootstrapService(
        persistence_service=persistence_service,
        session_factory=SessionLocal,
    )
    initializer = MCPRuntimeInitializer(
        loader=get_mcp_tool_loader(),
        tool_store=get_mcp_runtime_tool_store(),
    )
    return MCPLifecycleManager(
        runtime_initializer=initializer,
        connection_manager=get_mcp_connection_manager(),
        registry_bootstrapper=bootstrapper,
    )


def get_mcp_runtime_tools():
    """Return the startup-frozen MCP Tool snapshot for Agent composition."""

    return get_mcp_runtime_tool_store().snapshot()
