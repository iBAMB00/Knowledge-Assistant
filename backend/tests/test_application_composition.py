from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]


def test_main_keeps_mcp_lifespan_and_mounts_conversation_router() -> None:
    source = (BACKEND_ROOT / "app" / "main.py").read_text(
        encoding="utf-8"
    )

    assert "from contextlib import asynccontextmanager" in source
    assert "async def lifespan(" in source
    assert "get_mcp_lifecycle_manager" in source
    assert "await mcp_lifecycle.startup()" in source
    assert "await mcp_lifecycle.shutdown()" in source
    assert "lifespan=lifespan" in source
    assert "from app.api.conversation import router as conversation_router" in source
    assert "app.include_router(conversation_router)" in source


def test_database_model_exports_keep_mcp_server_and_conversation_models() -> None:
    source = (
        BACKEND_ROOT / "app" / "models" / "database" / "__init__.py"
    ).read_text(encoding="utf-8")

    assert "from app.models.database.mcp_server import MCPServer" in source
    assert "from app.models.database.conversation import Conversation" in source
    assert "ConversationMessage" in source
    assert '"MCPServer"' in source
    assert '"Conversation"' in source
    assert '"ConversationMessage"' in source
