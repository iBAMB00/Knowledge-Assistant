from sqlalchemy.orm import Session

from app.models.database.mcp_server import MCPServer


class MCPServerRepository:
    """MCP Server 持久化仓库，只负责数据库访问。"""

    def create(self, db: Session, server: MCPServer) -> MCPServer:
        db.add(server)
        db.flush()
        return server

    def find_by_server_id(
        self,
        db: Session,
        server_id: str,
    ) -> MCPServer | None:
        return (
            db.query(MCPServer)
            .filter(MCPServer.server_id == server_id)
            .first()
        )

    def list_enabled(self, db: Session) -> list[MCPServer]:
        return (
            db.query(MCPServer)
            .filter(MCPServer.enabled.is_(True))
            .all()
        )

    def update_status(
        self,
        db: Session,
        server: MCPServer,
        status: str,
    ) -> MCPServer:
        server.status = status
        db.flush()
        return server
