import json

from sqlalchemy.orm import Session

from app.agent.mcp.config import MCPServerConfig, MCPTransportType
from app.agent.mcp.registry import MCPServerRegistry
from app.models.database.mcp_server import MCPServer
from app.repositories.mcp_server_repository import MCPServerRepository


class MCPManagementService:
    """负责 MCP Server 持久化配置与 Runtime Registry 之间的同步。"""

    def __init__(
        self,
        repository: MCPServerRepository,
        registry: MCPServerRegistry,
    ) -> None:
        self.repository = repository
        self.registry = registry

    def restore_registry(self, db: Session) -> None:
        """从数据库恢复启用状态的 MCP Server 到 Runtime Registry。"""
        for server in self.repository.list_enabled(db):
            config = MCPServerConfig(
                server_id=server.server_id,
                transport=MCPTransportType(server.transport),
                command=server.command,
                args=json.loads(server.args_json),
                timeout_seconds=server.timeout_seconds,
            )
            self.registry.register(config)

    def create_server(
        self,
        db: Session,
        config: MCPServerConfig,
    ) -> MCPServer:
        server = MCPServer(
            server_id=config.server_id,
            transport=config.transport.value,
            command=config.command,
            args_json=json.dumps(config.args),
            timeout_seconds=config.timeout_seconds,
        )
        return self.repository.create(db, server)
