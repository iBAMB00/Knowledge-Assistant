from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from app.agent.context import ToolExecutionContext


class ToolRiskLevel(str, Enum):
    """Agent Tool 的最小风险等级。"""

    READ_ONLY = "read_only"


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


class ToolExecutionError(ToolError):
    """Tool 执行出现未预期内部错误。"""

    code = "execution_failed"


InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


class BaseAgentTool(ABC, Generic[InputT, OutputT]):
    """
    无框架依赖的 Agent Tool 基类。

    当前只承担稳定 Contract 与执行入口，
    不引入 Registry、LangChain、MCP 等运行时概念。
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

    @abstractmethod
    def execute(
        self,
        db: Session,
        context: ToolExecutionContext,
        tool_input: InputT,
    ) -> OutputT:
        """执行一次已经完成 Schema 校验的 Tool 调用。"""
