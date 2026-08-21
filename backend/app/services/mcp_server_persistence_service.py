from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.agent.mcp.config import MCPServerConfig, MCPTransportType
from app.agent.mcp.registry import MCPServerRegistry
from app.models.database.mcp_server import MCPServer
from app.repositories.mcp_server_repository import MCPServerRepository


class MCPServerConflictError(ValueError):
    """MCP Server 持久化配置冲突。"""


class MCPRegistryRestoreConflictError(RuntimeError):
    """数据库配置与当前 Runtime Registry 存在冲突。"""


class MCPServerPersistenceService:
    """维护 MCP Server 持久化配置与 Runtime Registry 的恢复边界。"""

    def __init__(
        self,
        *,
        repository: MCPServerRepository,
        registry: MCPServerRegistry,
    ) -> None:
        self._repository = repository
        self._registry = registry

    def create_server(
        self,
        db: Session,
        *,
        config: MCPServerConfig,
        enabled: bool = True,
    ) -> MCPServer:
        """保存 MCP Server 期望配置。

        当前小版本只建立持久化事实，不做运行时热加载；运行时恢复统一发生在
        MCP lifecycle startup，避免把 Dynamic Enable/Disable 偷渡进本轮范围。
        """
        server = MCPServer(
            server_id=config.server_id,
            transport=config.transport.value,
            command=config.command,
            args=list(config.args),
            timeout_seconds=config.timeout_seconds,
            enabled=enabled,
        )

        try:
            self._repository.create(db, server)
            db.commit()
            db.refresh(server)
            return server
        except IntegrityError as exc:
            db.rollback()
            raise MCPServerConflictError(
                f"MCP server already exists: {config.server_id}"
            ) from exc
        except Exception:
            db.rollback()
            raise

    def restore_registry(self, db: Session) -> list[MCPServerConfig]:
        """将数据库中 enabled 的 MCP Server 恢复到 Runtime Registry。

        恢复操作是幂等的：Registry 中已有完全相同配置时直接跳过；同一
        server_id 若配置不同则 fail closed，避免运行时静默覆盖配置。
        """
        restored: list[MCPServerConfig] = []

        for server in self._repository.find_enabled(db):
            config = self._to_runtime_config(server)
            existing = self._registry.get(config.server_id)

            if existing is None:
                self._registry.register(config)
                restored.append(config)
                continue

            if existing != config:
                raise MCPRegistryRestoreConflictError(
                    "MCP runtime registry conflicts with persisted config: "
                    f"{config.server_id}"
                )

        return restored

    @staticmethod
    def _to_runtime_config(server: MCPServer) -> MCPServerConfig:
        return MCPServerConfig(
            server_id=server.server_id,
            transport=MCPTransportType(server.transport),
            command=server.command,
            args=list(server.args or []),
            timeout_seconds=server.timeout_seconds,
        )
