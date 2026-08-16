import json
import sys
import types
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.agent.context import ToolExecutionContext
from app.agent.frameworks.langchain.tool_adapter import LangChainToolAdapter
from app.agent.tools.base import (
    BaseAgentTool,
    ToolResourceNotFoundError,
    ToolRiskLevel,
)
from app.constants.user_role import UserRole


class DemoInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)


class DemoOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str


class DemoTool(BaseAgentTool[DemoInput, DemoOutput]):
    name = "demo_search"
    version = "1.0.0"
    description = "Search demo data in the current trusted scope."
    risk_level = ToolRiskLevel.READ_ONLY
    input_model = DemoInput
    output_model = DemoOutput

    def __init__(self) -> None:
        self.last_context: ToolExecutionContext | None = None

    def execute(self, db, context, tool_input):
        self.last_context = context
        if tool_input.query == "missing":
            raise ToolResourceNotFoundError("demo resource not found")
        return DemoOutput(value=f"result:{tool_input.query}")


class EvidenceTool(DemoTool):
    name = "demo_evidence"

    def extract_evidence_refs(self, output: DemoOutput) -> list[str]:
        return ["doc:1:chunk:2", "doc:1:chunk:2"]


class FakeStructuredTool:
    """测试用最小 LangChain StructuredTool 替身。"""

    def __init__(
        self,
        *,
        func,
        name: str,
        description: str,
        args_schema,
        infer_schema: bool,
        handle_validation_error=None,
        **_kwargs,
    ) -> None:
        self.func = func
        self.name = name
        self.description = description
        self.args_schema = args_schema
        self.infer_schema = infer_schema
        self.handle_validation_error = handle_validation_error

    @classmethod
    def from_function(cls, **kwargs):
        return cls(**kwargs)

    def invoke(self, arguments: dict[str, Any]) -> str:
        try:
            validated = self.args_schema.model_validate(arguments)
        except ValidationError as exc:
            if self.handle_validation_error is None:
                raise
            return self.handle_validation_error(exc)
        return self.func(**validated.model_dump())


@pytest.fixture(autouse=True)
def fake_langchain_core(monkeypatch):
    tools_module = types.ModuleType("langchain_core.tools")
    tools_module.StructuredTool = FakeStructuredTool

    package = types.ModuleType("langchain_core")
    package.tools = tools_module

    monkeypatch.setitem(sys.modules, "langchain_core", package)
    monkeypatch.setitem(sys.modules, "langchain_core.tools", tools_module)


def build_context() -> ToolExecutionContext:
    return ToolExecutionContext(
        user_id=7,
        role=UserRole.USER,
        knowledge_base_id=11,
        request_id="req-v21-langchain",
        agent_run_id=101,
    )


def test_adapter_exports_core_contract_without_trusted_context_fields():
    core_tool = DemoTool()
    adapter = LangChainToolAdapter([core_tool])

    tools = adapter.bind_tools(db=object(), context=build_context())

    assert adapter.tool_names == ("demo_search",)
    assert len(tools) == 1
    assert tools[0].name == core_tool.name
    assert tools[0].description == core_tool.description
    assert tools[0].args_schema is DemoInput
    assert tools[0].infer_schema is False

    schema = tools[0].args_schema.model_json_schema()
    assert set(schema["properties"]) == {"query"}
    assert "user_id" not in schema["properties"]
    assert "knowledge_base_id" not in schema["properties"]
    assert "agent_run_id" not in schema["properties"]


def test_adapter_executes_through_dispatcher_with_bound_trusted_context():
    core_tool = DemoTool()
    context = build_context()
    tool = LangChainToolAdapter([core_tool]).bind_tools(
        db=object(),
        context=context,
    )[0]

    payload = json.loads(tool.invoke({"query": "qdrant"}))

    assert payload == {
        "ok": True,
        "result": {"value": "result:qdrant"},
    }
    assert core_tool.last_context == context


def test_adapter_rejects_model_attempt_to_inject_trusted_context():
    tool = LangChainToolAdapter([DemoTool()]).bind_tools(
        db=object(),
        context=build_context(),
    )[0]

    payload = json.loads(
        tool.invoke({"query": "qdrant", "user_id": 999})
    )

    assert payload == {
        "ok": False,
        "error": {
            "code": "invalid_arguments",
            "message": "invalid arguments for tool",
        },
    }


def test_adapter_preserves_safe_tool_error_semantics():
    tool = LangChainToolAdapter([DemoTool()]).bind_tools(
        db=object(),
        context=build_context(),
    )[0]

    payload = json.loads(tool.invoke({"query": "missing"}))

    assert payload == {
        "ok": False,
        "error": {
            "code": "resource_not_found",
            "message": "demo resource not found",
        },
    }


def test_adapter_exposes_normalized_evidence_refs_in_model_result():
    tool = LangChainToolAdapter([EvidenceTool()]).bind_tools(
        db=object(),
        context=build_context(),
    )[0]

    payload = json.loads(tool.invoke({"query": "qdrant"}))

    assert payload["ok"] is True
    assert payload["evidence_refs"] == ["doc:1:chunk:2"]


def test_adapter_requires_non_empty_tool_set():
    with pytest.raises(ValueError, match="tools cannot be empty"):
        LangChainToolAdapter([])


def test_adapter_serializes_parallel_tool_invocations_for_shared_db_session():
    """LangChain 并行分发时，同一请求绑定的 DB Session 仍保持串行 Tool 执行。"""

    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    class SlowTool(DemoTool):
        name = "slow_demo_search"

        def __init__(self) -> None:
            super().__init__()
            self._state_lock = threading.Lock()
            self.active = 0
            self.max_active = 0

        def execute(self, db, context, tool_input):
            with self._state_lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            try:
                time.sleep(0.03)
                return DemoOutput(value=f"result:{tool_input.query}")
            finally:
                with self._state_lock:
                    self.active -= 1

    core_tool = SlowTool()
    tool = LangChainToolAdapter([core_tool]).bind_tools(
        db=object(),
        context=build_context(),
    )[0]

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                lambda query: json.loads(tool.invoke({"query": query})),
                ["first", "second"],
            )
        )

    assert all(item["ok"] is True for item in results)
    assert core_tool.max_active == 1
