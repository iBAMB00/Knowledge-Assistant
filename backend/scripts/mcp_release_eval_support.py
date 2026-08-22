"""CLI-only helpers for running Agent Eval with the v2.2 MCP release probe loaded."""

from __future__ import annotations

import asyncio
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from app.agent.mcp.config import MCPServerConfig


RELEASE_PROBE_SERVER_ID = "release_probe"
RELEASE_PROBE_TOOL_NAME = "mcp__release_probe__echo"
RELEASE_PROBE_SERVER = Path("scripts/mcp_release_probe_server.py")


def start_release_probe_runtime() -> None:
    """Load the release-probe MCP server into the application Agent Tool store."""

    from app.api.dependencies.agent import reset_agent_runtime_caches
    from app.api.dependencies.mcp import (
        get_mcp_lifecycle_manager,
        get_mcp_server_registry,
    )

    registry = get_mcp_server_registry()
    if registry.get(RELEASE_PROBE_SERVER_ID) is None:
        registry.register(
            MCPServerConfig(
                server_id=RELEASE_PROBE_SERVER_ID,
                command=sys.executable,
                args=[str(RELEASE_PROBE_SERVER.resolve())],
            )
        )

    asyncio.run(get_mcp_lifecycle_manager().startup())
    reset_agent_runtime_caches()


def stop_release_probe_runtime() -> None:
    from app.api.dependencies.agent import reset_agent_runtime_caches
    from app.api.dependencies.mcp import get_mcp_lifecycle_manager

    asyncio.run(get_mcp_lifecycle_manager().shutdown())
    reset_agent_runtime_caches()


@contextmanager
def release_probe_runtime(enabled: bool) -> Iterator[None]:
    """Guarantee MCP release-probe cleanup even when Eval raises."""

    if not enabled:
        yield
        return

    start_release_probe_runtime()
    try:
        yield
    finally:
        stop_release_probe_runtime()
