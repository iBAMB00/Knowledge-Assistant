from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory


BACKEND_ROOT = Path(__file__).resolve().parents[1]


def _script_directory() -> ScriptDirectory:
    config = Config(str(BACKEND_ROOT / "alembic.ini"))
    config.set_main_option(
        "script_location",
        str(BACKEND_ROOT / "alembic"),
    )
    return ScriptDirectory.from_config(config)


def test_alembic_migration_graph_has_single_head() -> None:
    script = _script_directory()
    assert len(script.get_heads()) == 1


def test_conversation_migration_extends_mcp_registry_revision() -> None:
    script = _script_directory()
    conversation_revision = script.get_revision("5a9c1d7e3b42")

    assert conversation_revision is not None
    assert conversation_revision.down_revision == "43e6d9f2c1ab"


def test_checkpoint_migration_extends_conversation_revision() -> None:
    script = _script_directory()
    checkpoint_revision = script.get_revision("8b7d3c4e2a91")

    assert checkpoint_revision is not None
    assert checkpoint_revision.down_revision == "5a9c1d7e3b42"


def test_stateful_agent_run_status_migration_extends_checkpoint_revision() -> None:
    script = _script_directory()
    status_revision = script.get_revision("d6f4a8c2b913")

    assert status_revision is not None
    assert status_revision.down_revision == "8b7d3c4e2a91"


def test_conversation_summary_and_memory_extend_stateful_revision() -> None:
    script = _script_directory()
    summary_revision = script.get_revision("f4a1c9d8e2b7")
    memory_revision = script.get_revision("a6c5e4d3b2f1")

    assert summary_revision is not None
    assert summary_revision.down_revision == "d6f4a8c2b913"
    assert memory_revision is not None
    assert memory_revision.down_revision == "f4a1c9d8e2b7"
    assert script.get_heads() == ["a6c5e4d3b2f1"]
