from sqlalchemy.orm import Session

from app.models.database.mcp_server import MCPServer


class MCPServerRepository:
    """MCP Server 配置数据访问层；只查询、写入和 flush。"""

    def create(self, db: Session, server: MCPServer) -> MCPServer:
        db.add(server)
        db.flush()
        return server

    def find_by_server_id(self, db: Session, server_id: str) -> MCPServer | None:
        return (
            db.query(MCPServer)
            .filter(MCPServer.server_id == server_id)
            .one_or_none()
        )

    def find_enabled(self, db: Session) -> list[MCPServer]:
        return (
            db.query(MCPServer)
            .filter(MCPServer.enabled.is_(True))
            .order_by(MCPServer.id.asc())
            .all()
        )

    def find_all(self, db: Session) -> list[MCPServer]:
        return db.query(MCPServer).order_by(MCPServer.id.asc()).all()
