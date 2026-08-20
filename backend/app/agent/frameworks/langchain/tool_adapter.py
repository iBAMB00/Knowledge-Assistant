import json
import logging
from collections.abc import Sequence
from threading import Lock
from typing import Any
from uuid import uuid4

from sqlalchemy.orm import Session

from app.agent.context import ToolExecutionContext
from app.agent.model_response import LLMToolCall, LLMToolResult
from app.agent.tool_dispatcher import ToolDispatcher
from app.agent.tool_result_message import build_model_facing_tool_result_content
from app.agent.tools.base import BaseAgentTool, ToolError


logger = logging.getLogger(__name__)


class LangChainToolAdapter:
    """
    把无框架依赖的 BaseAgentTool 适配为 LangChain StructuredTool。

    设计边界：
    - LangChain 只拥有 Tool 的框架表示，不拥有业务执行逻辑；
    - 真正执行仍复用 ToolDispatcher，保持 Native Runtime 的参数校验、
      输出校验、权限上下文和 ToolError 语义；
    - db / ToolExecutionContext 在服务端按请求绑定，不进入模型可见 Schema。
    """

    ADAPTER_VERSION = "1.0.0"

    def __init__(
        self,
        tools: Sequence[BaseAgentTool[Any, Any]],
    ) -> None:
        if not tools:
            raise ValueError("tools cannot be empty")

        self._tools = list(tools)
        self._dispatcher = ToolDispatcher(self._tools)

    @property
    def tool_names(self) -> tuple[str, ...]:
        """返回当前 Adapter 暴露的 Tool 名称，便于测试和版本对比。"""

        return tuple(tool.name for tool in self._tools)

    def bind_tools(
        self,
        *,
        db: Session,
        context: ToolExecutionContext,
    ) -> list[Any]:
        """
        为一次受信任请求绑定 LangChain Tool。

        Session 和身份/知识库 Scope 都通过闭包注入，绝不能放入
        args_schema 让模型生成。返回值使用 Any 是为了让 Tool Core 在
        未安装 LangChain 时仍可被普通 Native Runtime 导入和测试。
        """

        StructuredTool = self._load_structured_tool()
        execution_lock = Lock()

        return [
            StructuredTool.from_function(
                func=self._build_bound_callable(
                    tool=tool,
                    db=db,
                    context=context,
                    execution_lock=execution_lock,
                ),
                name=tool.name,
                description=tool.description,
                args_schema=tool.get_model_input_schema(),
                infer_schema=False,
                handle_validation_error=self._handle_validation_error,
            )
            for tool in self._tools
        ]

    def _build_bound_callable(
        self,
        *,
        tool: BaseAgentTool[Any, Any],
        db: Session,
        context: ToolExecutionContext,
        execution_lock: Lock,
    ):
        """构建单个只绑定服务端可信上下文的 LangChain 调用函数。

        create_agent 可并行分发多个 Tool Call，但 SQLAlchemy Session 不应
        被多个 Tool 线程同时使用，因此同一请求的业务 Tool 先统一串行化。
        """

        def invoke_tool(**arguments: Any) -> str:
            with execution_lock:
                return self._dispatch_as_json(
                    tool=tool,
                    db=db,
                    context=context,
                    arguments=arguments,
                )

        return invoke_tool

    def _dispatch_as_json(
        self,
        *,
        tool: BaseAgentTool[Any, Any],
        db: Session,
        context: ToolExecutionContext,
        arguments: dict[str, Any],
    ) -> str:
        """
        将一次 LangChain Tool 调用桥接到现有 ToolDispatcher。

        返回格式刻意与 Native Agent 回填模型的 Tool Result 保持一致，
        这样后续对比 Native Loop / LangChain 时不会因为结果语义不同而
        污染 Eval。
        """

        tool_call = LLMToolCall(
            id=f"langchain-{uuid4().hex}",
            name=tool.name,
            arguments_json=json.dumps(
                arguments,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        )

        try:
            result = self._dispatcher.dispatch(
                db=db,
                context=context,
                tool_call=tool_call,
            )
        except ToolError as exc:
            payload: dict[str, Any] = {
                "ok": False,
                "error": {
                    "code": exc.code,
                    "message": str(exc),
                },
            }
        else:
            payload = {
                "ok": True,
                "result": result.output,
            }
            if result.evidence_refs:
                payload["evidence_refs"] = result.evidence_refs

        raw_content = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return build_model_facing_tool_result_content(
            LLMToolResult(
                call_id=tool_call.id,
                tool_name=tool.name,
                content_json=raw_content,
            )
        )

    @staticmethod
    def _handle_validation_error(_exc: Exception) -> str:
        """把 LangChain 前置 Schema 校验失败转成稳定、安全的 Tool 结果。"""

        return json.dumps(
            {
                "ok": False,
                "error": {
                    "code": "invalid_arguments",
                    "message": "invalid arguments for tool",
                },
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @staticmethod
    def _load_structured_tool():
        """延迟加载 LangChain，避免 Native Agent 路径产生框架硬依赖。"""

        try:
            from langchain_core.tools import StructuredTool
        except ImportError as exc:
            raise RuntimeError(
                "LangChain Tool Adapter requires langchain-core; "
                "install project requirements before using v2.1 framework integration"
            ) from exc

        return StructuredTool
