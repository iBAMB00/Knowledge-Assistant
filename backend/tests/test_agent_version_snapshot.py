from app.agent.tools.base import ToolContract, ToolRiskLevel
from app.agent.version_snapshot import (
    AGENT_RUNTIME_VERSION,
    build_agent_runtime_version_snapshot,
    build_retrieval_config_version,
    build_toolset_version,
)
from app.core.config import Settings


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "model_provider": "test",
        "model_base_url": "http://model-a",
        "model_name": "agent-model",
        "model_api_key": "secret-model-a",
        "embedding_provider": "test-embedding",
        "embedding_base_url": "http://embedding-a",
        "embedding_model": "embedding-v1",
        "embedding_api_key": "secret-embedding-a",
        "jwt_secret_key": "12345678901234567890123456789012",
        "vector_store_backend": "qdrant",
        "qdrant_url": "http://qdrant-a:6333",
        "qdrant_collection_name": "chunks",
    }
    values.update(overrides)
    return Settings(**values)


def _tool_contract(
    *,
    name: str,
    version: str,
    description: str = "tool",
) -> ToolContract:
    return ToolContract(
        name=name,
        version=version,
        description=description,
        risk_level=ToolRiskLevel.READ_ONLY,
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
        },
        output_schema={"type": "object"},
    )


def test_toolset_version_is_order_independent_and_contract_sensitive() -> None:
    first = _tool_contract(name="search", version="1.0.0")
    second = _tool_contract(name="list", version="2.0.0")

    version_a = build_toolset_version([first, second])
    version_b = build_toolset_version([second, first])
    changed = build_toolset_version(
        [
            first,
            _tool_contract(
                name="list",
                version="2.0.0",
                description="changed contract",
            ),
        ]
    )

    assert version_a == version_b
    assert version_a.startswith("toolset-v1:")
    assert changed != version_a


def test_retrieval_config_version_tracks_behavior_not_secrets_or_endpoints() -> None:
    base = build_retrieval_config_version(_settings())
    secret_and_endpoint_only = build_retrieval_config_version(
        _settings(
            model_api_key="another-secret",
            embedding_api_key="another-embedding-secret",
            qdrant_url="http://qdrant-b:6333",
        )
    )
    changed_behavior = build_retrieval_config_version(
        _settings(retrieval_rrf_k=80)
    )

    assert base == secret_and_endpoint_only
    assert base.startswith("retrieval-v1:")
    assert changed_behavior != base


def test_runtime_snapshot_combines_manual_and_automatic_versions() -> None:
    snapshot = build_agent_runtime_version_snapshot(
        settings=_settings(),
        tool_contracts=[_tool_contract(name="search", version="1.0.0")],
        prompt_version="1.0.0",
    )

    assert snapshot.agent_version == AGENT_RUNTIME_VERSION
    assert snapshot.prompt_version == "1.0.0"
    assert snapshot.toolset_version.startswith("toolset-v1:")
    assert snapshot.retrieval_config_version.startswith("retrieval-v1:")


def test_runtime_snapshot_can_identify_framework_candidate_version() -> None:
    snapshot = build_agent_runtime_version_snapshot(
        settings=_settings(),
        tool_contracts=[_tool_contract(name="search", version="1.0.0")],
        prompt_version="1.0.0",
        agent_version="langchain-v1:1.3.0",
    )

    assert snapshot.agent_version == "langchain-v1:1.3.0"


def test_runtime_snapshot_rejects_empty_agent_version_override() -> None:
    import pytest

    with pytest.raises(ValueError, match="agent_version cannot be empty"):
        build_agent_runtime_version_snapshot(
            settings=_settings(),
            tool_contracts=[_tool_contract(name="search", version="1.0.0")],
            prompt_version="1.0.0",
            agent_version="   ",
        )
