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
