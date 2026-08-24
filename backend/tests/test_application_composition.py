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
    assert "ConversationSummary" in source
    assert '"MCPServer"' in source
    assert '"Conversation"' in source
    assert '"ConversationMessage"' in source
    assert '"ConversationSummary"' in source


def test_database_model_exports_agent_thread_and_checkpoint() -> None:
    source = (
        BACKEND_ROOT / "app" / "models" / "database" / "__init__.py"
    ).read_text(encoding="utf-8")

    assert "AgentThread" in source
    assert "AgentCheckpoint" in source
    assert '"AgentThread"' in source
    assert '"AgentCheckpoint"' in source

def test_agent_runtime_composition_injects_shared_conversation_context_provider(
) -> None:
    source = (
        BACKEND_ROOT
        / "app"
        / "api"
        / "dependencies"
        / "agent.py"
    ).read_text(encoding="utf-8")

    assert "def get_conversation_history_context_provider(" in source
    assert "def get_conversation_summary_service(" in source
    assert "def get_conversation_context_provider(" in source
    assert source.count("get_conversation_context_provider()") == 4


def test_b7_memory_extraction_is_wired_after_successful_agent_messages() -> None:
    dependencies_source = (
        BACKEND_ROOT / "app" / "api" / "dependencies" / "agent.py"
    ).read_text(encoding="utf-8")
    api_source = (
        BACKEND_ROOT / "app" / "api" / "agent.py"
    ).read_text(encoding="utf-8")

    assert "def get_conversation_memory_extraction_service(" in dependencies_source
    assert "LLMConversationMemoryExtractor(" in dependencies_source
    assert "def _extract_completed_turn_memory(" in api_source
    assert "service.extract_completed_turn(" in api_source


def test_b8_memory_retrieval_is_composed_into_shared_conversation_context() -> None:
    source = (
        BACKEND_ROOT / "app" / "api" / "dependencies" / "agent.py"
    ).read_text(encoding="utf-8")

    assert "def get_conversation_memory_context_provider(" in source
    assert "LexicalConversationMemoryRetriever(" in source
    assert "memory_context_provider=get_conversation_memory_context_provider()" in source
    assert "get_conversation_memory_context_provider.cache_clear()" in source


def test_b9_context_usage_is_wired_through_all_agent_execution_bridges() -> None:
    service_paths = [
        BACKEND_ROOT / "app" / "services" / "agent_execution_service.py",
        BACKEND_ROOT / "app" / "services" / "langchain_agent_execution_service.py",
        BACKEND_ROOT / "app" / "services" / "langgraph_agent_execution_service.py",
    ]

    for path in service_paths:
        source = path.read_text(encoding="utf-8")
        assert "resolve_agent_context_usage" in source
        assert 'update={"context_usage": context_usage}' in source
