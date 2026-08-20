from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy.orm import Session

from app.agent.context import ToolExecutionContext


class ToolRiskLevel(str, Enum):
    """Agent Tool 的最小风险等级。"""

    READ_ONLY = "read_only"


class ToolSource(str, Enum):
    """Tool Contract 的能力来源；用于版本快照与运行时诊断。"""

    LOCAL = "local"
    MCP = "mcp"


class ToolContract(BaseModel):
    """Agent Runtime 可消费的稳定 Tool 元数据。"""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    description: str = Field(min_length=1)
    risk_level: ToolRiskLevel
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    source: ToolSource = ToolSource.LOCAL
    source_id: str | None = Field(default=None, min_length=1, max_length=64)

    @model_validator(mode="after")
    def validate_source(self) -> "ToolContract":
        """MCP Contract 必须携带稳定本地 server id；Local Contract 不需要。"""

        if self.source is ToolSource.MCP and self.source_id is None:
            raise ValueError("MCP tool contract requires source_id")
        if self.source is ToolSource.LOCAL and self.source_id is not None:
            raise ValueError("local tool contract cannot define source_id")
        return self


class ToolError(RuntimeError):
    """Tool 层可识别的基础异常。"""

    code = "tool_error"
    retryable = False


class ToolInvalidArgumentsError(ToolError):
    """Tool 参数或底层业务参数非法。"""

    code = "invalid_arguments"


class ToolResourceNotFoundError(ToolError):
    """资源不存在，或当前调用主体无权知道资源存在。"""

    code = "resource_not_found"


class ToolNotFoundError(ToolError):
    """模型请求了当前 Runtime 未注册的 Tool。"""

    code = "tool_not_found"


class ToolExecutionError(ToolError):
    """Tool 执行出现未预期内部错误。"""

    code = "execution_failed"


InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


class BaseAgentTool(ABC, Generic[InputT, OutputT]):
    """
    无框架依赖的 Agent Tool 基类。

    当前只承担稳定 Contract、Schema hook 与执行入口，
    不依赖 Registry、LangChain 或具体远端 Transport。
    """

    name: str
    version: str
    description: str
    risk_level: ToolRiskLevel
    input_model: type[InputT]
    output_model: type[OutputT]

    def get_contract(self) -> ToolContract:
        """根据 Pydantic Input / Output 生成结构化 Tool Contract。"""

        return ToolContract(
            name=self.name,
            version=self.version,
            description=self.description,
            risk_level=self.risk_level,
            input_schema=self.input_model.model_json_schema(),
            output_schema=self.output_model.model_json_schema(),
        )

    def get_model_input_schema(self) -> type[BaseModel] | dict[str, Any]:
        """返回模型/Framework 可见的参数 Schema；动态 Tool 可覆盖为 JSON Schema。"""

        return self.input_model

    def validate_input(self, arguments: dict[str, Any]) -> InputT:
        """校验不可信模型参数；Local Tool 默认使用 Pydantic Input Model。"""

        return self.input_model.model_validate(arguments)

    def validate_output(self, raw_output: Any) -> OutputT:
        """校验 Tool 输出；Local Tool 默认使用 Pydantic Output Model。"""

        return self.output_model.model_validate(raw_output)

    def extract_evidence_refs(self, output: OutputT) -> list[str]:
        """提取可安全用于 Eval / Citation 的来源引用；普通 Tool 默认没有证据引用。"""

        return []

    @abstractmethod
    def execute(
        self,
        db: Session,
        context: ToolExecutionContext,
        tool_input: InputT,
    ) -> OutputT:
        """执行一次已经完成 Schema 校验的 Tool 调用。"""
