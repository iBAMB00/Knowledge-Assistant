import json
import logging
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy.orm import Session

from app.agent.context import ToolExecutionContext
from app.agent.model_response import LLMToolCall
from app.agent.tools.base import (
    BaseAgentTool,
    ToolError,
    ToolExecutionError,
    ToolInvalidArgumentsError,
    ToolNotFoundError,
)


logger = logging.getLogger(__name__)


class ToolDispatchResult(BaseModel):
    """一次 Tool Dispatch 的稳定结构化结果。"""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    call_id: str
    tool_name: str
    output: dict[str, Any]
    evidence_refs: list[str] = Field(default_factory=list)


class ToolDispatcher:
    """
    Native Agent Runtime 的最小 Tool Dispatcher。

    只负责：
    LLMToolCall -> Tool 查找 -> JSON 解析 -> Pydantic 校验
    -> Tool.execute() -> Output Contract 校验。

    不负责模型循环、重试、预算、事务提交或 HTTP 异常映射。
    """

    def __init__(
        self,
        tools: Sequence[BaseAgentTool[Any, Any]],
    ) -> None:
        if not tools:
            raise ValueError("tools cannot be empty")

        self._tools: dict[str, BaseAgentTool[Any, Any]] = {}

        for tool in tools:
            if tool.name in self._tools:
                raise ValueError(
                    f"duplicate tool name: {tool.name}"
                )

            self._tools[tool.name] = tool

    def dispatch(
        self,
        *,
        db: Session,
        context: ToolExecutionContext,
        tool_call: LLMToolCall,
    ) -> ToolDispatchResult:
        """校验并执行模型请求的一次 Tool Call。"""

        tool = self._tools.get(tool_call.name)

        if tool is None:
            raise ToolNotFoundError(
                f"tool not found: {tool_call.name}"
            )

        arguments = self._parse_arguments(tool_call)
        tool_input = self._validate_input(
            tool=tool,
            arguments=arguments,
        )

        logger.info(
            "Tool dispatch started: request_id=%s "
            "tool_name=%s call_id=%s",
            context.request_id,
            tool.name,
            tool_call.id,
        )

        try:
            raw_output = tool.execute(
                db=db,
                context=context,
                tool_input=tool_input,
            )
        except ToolError:
            raise
        except Exception as exc:
            logger.error(
                "Tool dispatch failed: request_id=%s "
                "tool_name=%s call_id=%s error_type=%s",
                context.request_id,
                tool.name,
                tool_call.id,
                type(exc).__name__,
            )
            raise ToolExecutionError(
                f"tool execution failed: {tool.name}"
            ) from exc

        try:
            validated_output = tool.output_model.model_validate(
                raw_output
            )
        except ValidationError as exc:
            logger.error(
                "Tool output validation failed: request_id=%s "
                "tool_name=%s call_id=%s error_count=%d",
                context.request_id,
                tool.name,
                tool_call.id,
                exc.error_count(),
            )
            raise ToolExecutionError(
                f"invalid tool output: {tool.name}"
            ) from exc

        logger.info(
            "Tool dispatch completed: request_id=%s "
            "tool_name=%s call_id=%s",
            context.request_id,
            tool.name,
            tool_call.id,
        )

        evidence_refs = self._normalize_evidence_refs(
            tool.extract_evidence_refs(validated_output)
        )

        return ToolDispatchResult(
            call_id=tool_call.id,
            tool_name=tool.name,
            output=validated_output.model_dump(mode="json"),
            evidence_refs=evidence_refs,
        )


    @staticmethod
    def _normalize_evidence_refs(values: list[str]) -> list[str]:
        """过滤空引用并保持顺序去重，避免观察链收到脏元数据。"""

        refs: list[str] = []
        seen: set[str] = set()

        for value in values:
            normalized = value.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            refs.append(normalized)

        return refs

    @staticmethod
    def _parse_arguments(
        tool_call: LLMToolCall,
    ) -> dict[str, Any]:
        """把模型原始 arguments JSON 解析为对象。"""

        try:
            arguments = json.loads(tool_call.arguments_json)
        except json.JSONDecodeError as exc:
            raise ToolInvalidArgumentsError(
                f"invalid JSON arguments for tool: {tool_call.name}"
            ) from exc

        if not isinstance(arguments, dict):
            raise ToolInvalidArgumentsError(
                f"tool arguments must be an object: {tool_call.name}"
            )

        return arguments

    @staticmethod
    def _validate_input(
        *,
        tool: BaseAgentTool[Any, Any],
        arguments: dict[str, Any],
    ) -> BaseModel:
        """使用 Tool 自己的 Input Model 校验不可信模型参数。"""

        try:
            return tool.input_model.model_validate(arguments)
        except ValidationError as exc:
            raise ToolInvalidArgumentsError(
                f"invalid arguments for tool: {tool.name}"
            ) from exc
