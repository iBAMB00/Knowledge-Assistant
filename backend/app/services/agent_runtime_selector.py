"""Agent HTTP 入口的同步 Runtime 选择器。"""

from collections.abc import Callable
from typing import Protocol, TypeAlias

from sqlalchemy.orm import Session

from app.agent.context import ToolExecutionContext
from app.agent.frameworks.langchain.runner import LangChainAgentResult
from app.agent.native_agent import NativeAgentResult
from app.constants.agent_runtime import AgentRuntime


AgentExecutionResult: TypeAlias = NativeAgentResult | LangChainAgentResult


class SynchronousAgentExecutionService(Protocol):
    """Native / Framework 执行服务共同满足的最小同步接口。"""

    def run(
        self,
        *,
        db: Session,
        context: ToolExecutionContext,
        message: str,
    ) -> AgentExecutionResult:
        """执行一次同步 Agent Run。"""
        ...


class AgentRuntimeUnavailableError(RuntimeError):
    """请求了当前部署未开放的 Agent Runtime。"""


class AgentRuntimeSelector:
    """
    为 HTTP 请求按需选择 Native 或 LangChain Candidate。

    使用 factory 而不是预先构造两个 Runtime，保证默认 Native 请求不会
    因为 Candidate 未启用而初始化 LangChain Model / Framework 依赖。
    """

    def __init__(
        self,
        *,
        native_factory: Callable[[], SynchronousAgentExecutionService],
        langchain_factory: Callable[[], SynchronousAgentExecutionService],
        langchain_candidate_enabled: bool,
    ) -> None:
        self._native_factory = native_factory
        self._langchain_factory = langchain_factory
        self._langchain_candidate_enabled = langchain_candidate_enabled

    def select(
        self,
        runtime: AgentRuntime,
    ) -> SynchronousAgentExecutionService:
        """返回请求对应的同步执行服务；Candidate 未开放时显式拒绝。"""

        if runtime == AgentRuntime.NATIVE:
            return self._native_factory()

        if runtime == AgentRuntime.LANGCHAIN:
            if not self._langchain_candidate_enabled:
                raise AgentRuntimeUnavailableError(
                    "langchain candidate runtime is disabled"
                )
            return self._langchain_factory()

        raise AgentRuntimeUnavailableError(
            f"unsupported agent runtime: {runtime}"
        )
