from collections.abc import Callable

from sqlalchemy.orm import Session

from app.services.mcp_server_persistence_service import MCPServerPersistenceService


class MCPRegistryBootstrapService:
    """在应用生命周期启动时，用独立数据库会话恢复 MCP Runtime Registry。"""

    def __init__(
        self,
        *,
        persistence_service: MCPServerPersistenceService,
        session_factory: Callable[[], Session],
    ) -> None:
        self._persistence_service = persistence_service
        self._session_factory = session_factory

    def restore(self) -> None:
        db = self._session_factory()
        try:
            self._persistence_service.restore_registry(db)
        finally:
            db.close()
