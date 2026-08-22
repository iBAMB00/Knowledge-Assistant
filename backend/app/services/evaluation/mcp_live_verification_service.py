from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from sqlalchemy.orm import Session

from app.agent.context import ToolExecutionContext
from app.agent.mcp.client import MCPRegistryInvoker
from app.agent.mcp.config import MCPServerConfig
from app.agent.mcp.connection import MCPConnectionManager
from app.agent.mcp.discovery import MCPToolDiscoveryService
from app.agent.mcp.loader import MCPToolLoader
from app.agent.mcp.namespace import MCPToolNamespaceRegistry
from app.agent.mcp.registry import MCPServerRegistry
from app.agent.mcp.runtime import MCPToolRuntimeBuilder
from app.agent.mcp.stdio_transport import StdioMCPTransportAdapter
from app.agent.mcp.transport import MCPClientSessionManager
from app.agent.model_response import LLMToolCall
from app.agent.tool_dispatcher import ToolDispatcher
from app.agent.version_snapshot import build_toolset_version
from app.constants.user_role import UserRole
from app.schemas.mcp_release import MCPLiveVerificationReport


class MCPLiveVerificationService:
    """通过官方 SDK + stdio + ToolDispatcher 做真实 MCP 发布探针。"""

    VERIFIER_VERSION = "1.0.0"
    SERVER_ID = "release_probe"
    REMOTE_TOOL_NAME = "echo"
    PROBE_MESSAGE = "v2.2-release-probe"

    def run(self, *, server_script: Path) -> MCPLiveVerificationReport:
        config = MCPServerConfig(
            server_id=self.SERVER_ID,
            command=sys.executable,
            args=[str(server_script.resolve())],
        )
        registry = MCPServerRegistry()
        registry.register(config)

        connection_manager = MCPConnectionManager(
            registry=registry,
            session_factory=lambda item: MCPClientSessionManager(
                StdioMCPTransportAdapter(item)
            ),
        )
        invoker = MCPRegistryInvoker(connection_manager=connection_manager)
        loader = MCPToolLoader(
            registry=registry,
            invoker=invoker,
            namespace_registry=MCPToolNamespaceRegistry(),
            runtime_builder=MCPToolRuntimeBuilder(
                MCPToolDiscoveryService(invoker=invoker)
            ),
        )

        exposed_name = f"mcp__{self.SERVER_ID}__{self.REMOTE_TOOL_NAME}"
        discovery_succeeded = False
        dispatch_succeeded = False
        structured_output_succeeded = False
        result_echo: str | None = None
        tool_contract_version = "unknown"
        mcp_toolset_version = "unknown"
        failure_reason: str | None = None

        try:
            tools = asyncio.run(loader.load_tools())
            discovered = {tool.name: tool for tool in tools}
            tool = discovered.get(exposed_name)
            if tool is None:
                raise RuntimeError(f"MCP release probe tool not discovered: {exposed_name}")
            discovery_succeeded = True

            contract = tool.get_contract()
            tool_contract_version = contract.version
            mcp_toolset_version = build_toolset_version(
                [item.get_contract() for item in tools]
            )

            dispatcher = ToolDispatcher(tools)
            db = Session()
            try:
                result = dispatcher.dispatch(
                    db=db,
                    context=ToolExecutionContext(
                        user_id=1,
                        role=UserRole.ADMIN,
                        knowledge_base_id=1,
                        request_id="mcp-release-live-verification",
                    ),
                    tool_call=LLMToolCall(
                        id="mcp-release-probe-call",
                        name=exposed_name,
                        arguments_json=json.dumps(
                            {"message": self.PROBE_MESSAGE},
                            ensure_ascii=False,
                        ),
                    ),
                )
            finally:
                db.close()

            dispatch_succeeded = True
            structured = result.output.get("structured_content")
            if isinstance(structured, dict):
                value = structured.get("result")
                if isinstance(value, str):
                    result_echo = value
            structured_output_succeeded = result_echo == self.PROBE_MESSAGE
            if not structured_output_succeeded:
                raise RuntimeError("MCP release probe structured output mismatch")

        except Exception as exc:
            failure_reason = f"{type(exc).__name__}: {exc}"
        finally:
            asyncio.run(connection_manager.disconnect_all())

        decision = (
            "pass"
            if discovery_succeeded
            and dispatch_succeeded
            and structured_output_succeeded
            else "fail"
        )

        try:
            sdk_version = version("mcp")
        except PackageNotFoundError:
            sdk_version = "not-installed"

        return MCPLiveVerificationReport(
            generated_at=datetime.now(timezone.utc),
            verifier_version=self.VERIFIER_VERSION,
            sdk_version=sdk_version,
            transport="stdio",
            server_id=self.SERVER_ID,
            remote_tool_name=self.REMOTE_TOOL_NAME,
            exposed_tool_name=exposed_name,
            tool_contract_version=tool_contract_version,
            mcp_toolset_version=mcp_toolset_version,
            discovery_succeeded=discovery_succeeded,
            dispatch_succeeded=dispatch_succeeded,
            structured_output_succeeded=structured_output_succeeded,
            result_echo=result_echo,
            decision=decision,
            failure_reason=failure_reason,
        )
