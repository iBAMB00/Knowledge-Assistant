import hashlib
import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


_MCP_SERVER_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
_MCP_SAFE_TOOL_NAME_PATTERN = re.compile(r"[^A-Za-z0-9_-]")
_MODEL_TOOL_NAME_MAX_LENGTH = 64
_TRUSTED_SCOPE_FIELDS = frozenset(
    {
        "user_id",
        "role",
        "knowledge_base_id",
        "request_id",
        "agent_run_id",
    }
)


class MCPToolAnnotations(BaseModel):
    """MCP Server 声明的 Tool hints；只保留为观测元数据，不作为本地安全策略。"""

    model_config = ConfigDict(frozen=True, extra="ignore")

    read_only_hint: bool | None = None
    destructive_hint: bool | None = None
    idempotent_hint: bool | None = None
    open_world_hint: bool | None = None


class MCPRemoteToolDescriptor(BaseModel):
    """一次 MCP tools/list 发现后，进入 Agent Core 前的稳定远端 Tool 描述。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    server_id: str = Field(min_length=1, max_length=32)
    remote_name: str = Field(min_length=1, max_length=128)
    title: str | None = Field(default=None, max_length=200)
    description: str | None = Field(default=None, max_length=4000)
    input_schema: dict[str, Any]
    output_schema: dict[str, Any] | None = None
    annotations: MCPToolAnnotations = Field(default_factory=MCPToolAnnotations)

    @model_validator(mode="after")
    def validate_boundary(self) -> "MCPRemoteToolDescriptor":
        """拒绝不稳定 server alias、非对象参数以及模型可注入的 Trusted Context。"""

        if _MCP_SERVER_ID_PATTERN.fullmatch(self.server_id) is None:
            raise ValueError(
                "server_id must contain only letters, digits, '_' or '-'"
            )

        schema_type = self.input_schema.get("type")
        if schema_type != "object":
            raise ValueError("MCP tool input schema root must be an object")

        properties = self.input_schema.get("properties", {})
        if not isinstance(properties, dict):
            raise ValueError("MCP tool input schema properties must be an object")

        leaked_scope = _TRUSTED_SCOPE_FIELDS.intersection(properties)
        if leaked_scope:
            raise ValueError(
                "MCP tool input schema cannot expose trusted context fields: "
                + ", ".join(sorted(leaked_scope))
            )

        if self.output_schema is not None:
            output_type = self.output_schema.get("type")
            if output_type != "object":
                raise ValueError("MCP tool output schema root must be an object")

        return self

    @property
    def exposed_name(self) -> str:
        """返回在聚合 Agent Toolset 中稳定且避免跨 Server 冲突的名称。"""

        return build_mcp_exposed_tool_name(
            server_id=self.server_id,
            remote_name=self.remote_name,
        )

    @property
    def contract_version(self) -> str:
        """MCP 协议没有 Tool version 字段，因此对模型可见 Contract 生成稳定版本。"""

        payload = {
            "remote_name": self.remote_name,
            "title": self.title,
            "description": self.description,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()[:16]
        return f"mcp-v1:{digest}"


class MCPToolCallResult(BaseModel):
    """Transport / SDK 层归一化后交给 Agent Core 的最小 MCP Tool Result。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    is_error: bool = False
    structured_content: dict[str, Any] | None = None
    text_content: list[str] = Field(default_factory=list)

    @classmethod
    def from_sdk_result(cls, result: Any) -> "MCPToolCallResult":
        """把官方 Python SDK ``CallToolResult`` 收敛到稳定 Host Contract。"""

        text_content: list[str] = []
        for block in getattr(result, "content", []) or []:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                text_content.append(text)

        # MCP Python SDK v1.x generated models expose protocol field names
        # (structuredContent / isError), while newer SDK surfaces may expose
        # Pythonic snake_case attributes. Accept both at the SDK boundary and
        # normalize them into our framework-neutral Host Contract.
        structured_content = getattr(result, "structured_content", None)
        if structured_content is None:
            structured_content = getattr(result, "structuredContent", None)
        if structured_content is not None and not isinstance(
            structured_content, dict
        ):
            structured_content = None

        is_error = getattr(result, "is_error", None)
        if is_error is None:
            is_error = getattr(result, "isError", False)

        return cls(
            is_error=bool(is_error),
            structured_content=structured_content,
            text_content=text_content,
        )


def build_mcp_exposed_tool_name(*, server_id: str, remote_name: str) -> str:
    """为多 MCP Server 聚合生成模型兼容、稳定、可消歧的 Tool 名称。"""

    if _MCP_SERVER_ID_PATTERN.fullmatch(server_id) is None:
        raise ValueError("invalid MCP server_id")

    normalized_remote = _MCP_SAFE_TOOL_NAME_PATTERN.sub("_", remote_name).strip("_")
    if not normalized_remote:
        normalized_remote = "tool"

    prefix = f"mcp__{server_id}__"
    candidate = f"{prefix}{normalized_remote}"

    needs_hash = normalized_remote != remote_name or len(candidate) > _MODEL_TOOL_NAME_MAX_LENGTH
    if not needs_hash:
        return candidate

    digest = hashlib.sha256(remote_name.encode("utf-8")).hexdigest()[:8]
    suffix = f"__{digest}"
    available = _MODEL_TOOL_NAME_MAX_LENGTH - len(prefix) - len(suffix)
    if available < 1:
        raise ValueError("MCP server_id is too long for model tool naming")

    trimmed_remote = normalized_remote[:available]
    return f"{prefix}{trimmed_remote}{suffix}"
