from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError as JSONSchemaValidationError
from jsonschema.validators import validator_for
from pydantic import BaseModel, ConfigDict, Field, RootModel
from sqlalchemy.orm import Session

from app.agent.context import ToolExecutionContext
from app.agent.mcp.client import MCPToolInvoker
from app.agent.mcp.contracts import MCPRemoteToolDescriptor
from app.agent.tools.base import (
    BaseAgentTool,
    ToolContract,
    ToolExecutionError,
    ToolInvalidArgumentsError,
    ToolRiskLevel,
    ToolSource,
)


class MCPToolArguments(RootModel[dict[str, Any]]):
    """通过远端 JSON Schema 校验后的动态 MCP Tool 参数。"""


class MCPToolOutput(BaseModel):
    """进入现有 ToolDispatcher 后的安全 MCP 输出 Envelope。"""

    model_config = ConfigDict(extra="forbid")

    structured_content: dict[str, Any] | None = None
    text_content: list[str] = Field(default_factory=list)


class MCPRemoteToolError(ToolExecutionError):
    """MCP Server 明确返回 isError 时使用的稳定 ToolError。"""

    code = "mcp_tool_error"


class MCPBackedAgentTool(BaseAgentTool[MCPToolArguments, MCPToolOutput]):
    """
    把一个已发现、已本地批准的 MCP Tool 适配进现有 BaseAgentTool 边界。

    注意：
    - MCP annotations 不决定风险等级；risk_level 必须由本地注册策略显式提供；
    - Trusted Context 通过 MCPToolInvoker 独立传递，不进入模型参数 Schema；
    - A1 不负责真实 MCP transport / session，只定义可测试的执行边界。
    """

    input_model = MCPToolArguments
    output_model = MCPToolOutput

    def __init__(
        self,
        *,
        descriptor: MCPRemoteToolDescriptor,
        invoker: MCPToolInvoker,
        approved_risk_level: ToolRiskLevel,
    ) -> None:
        if approved_risk_level is not ToolRiskLevel.READ_ONLY:
            raise ValueError("v2.2-A1 only permits locally approved read-only MCP tools")

        self.descriptor = descriptor
        self.invoker = invoker
        self.name = descriptor.exposed_name
        self.version = descriptor.contract_version
        self.description = (
            descriptor.description
            or descriptor.title
            or f"MCP tool {descriptor.remote_name}"
        )
        self.risk_level = approved_risk_level

        try:
            self._input_validator = self._build_validator(
                descriptor.input_schema
            )
            self._output_validator = (
                None
                if descriptor.output_schema is None
                else self._build_validator(descriptor.output_schema)
            )
        except SchemaError as exc:
            raise ValueError("invalid MCP tool JSON Schema") from exc

    def get_contract(self) -> ToolContract:
        """把远端 Tool 描述映射为项目现有稳定 ToolContract。"""

        return ToolContract(
            name=self.name,
            version=self.version,
            description=self.description,
            risk_level=self.risk_level,
            input_schema=self.descriptor.input_schema,
            output_schema=(
                self.descriptor.output_schema
                or MCPToolOutput.model_json_schema()
            ),
            source=ToolSource.MCP,
            source_id=self.descriptor.server_id,
        )

    def get_model_input_schema(self) -> dict[str, Any]:
        """直接向 Runtime 暴露 MCP tools/list 返回的 JSON Schema。"""

        return self.descriptor.input_schema

    def validate_input(self, arguments: dict[str, Any]) -> MCPToolArguments:
        """在远端调用发生前由 Host 使用 MCP JSON Schema 再校验一次模型参数。"""

        try:
            self._input_validator.validate(arguments)
        except JSONSchemaValidationError as exc:
            raise ToolInvalidArgumentsError(
                f"invalid arguments for tool: {self.name}"
            ) from exc

        return MCPToolArguments(arguments)

    def execute(
        self,
        db: Session,
        context: ToolExecutionContext,
        tool_input: MCPToolArguments,
    ) -> MCPToolOutput:
        """通过 MCPToolInvoker 执行远端 Tool，并把结果收敛到安全 Envelope。"""

        del db  # MCP transport 不直接拥有本地数据库 Session。

        result = self.invoker.call_tool(
            server_id=self.descriptor.server_id,
            tool_name=self.descriptor.remote_name,
            arguments=dict(tool_input.root),
            context=context,
        )

        if result.is_error:
            raise MCPRemoteToolError("remote MCP tool execution failed")

        if self._output_validator is not None:
            if result.structured_content is None:
                raise ToolExecutionError(
                    f"missing MCP structured output: {self.name}"
                )
            try:
                self._output_validator.validate(result.structured_content)
            except JSONSchemaValidationError as exc:
                raise ToolExecutionError(
                    f"invalid MCP structured output: {self.name}"
                ) from exc

        return MCPToolOutput(
            structured_content=result.structured_content,
            text_content=list(result.text_content),
        )

    @staticmethod
    def _build_validator(schema: dict[str, Any]):
        """遵循 MCP JSON Schema 声明；未指定 draft 时默认使用 2020-12。"""

        validator_cls = (
            Draft202012Validator
            if "$schema" not in schema
            else validator_for(schema)
        )
        validator_cls.check_schema(schema)
        return validator_cls(schema)
