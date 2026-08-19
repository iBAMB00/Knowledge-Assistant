"""Agent HTTP 同步 / SSE 入口的 Runtime 选择器。"""

from collections.abc import Callable, Iterator
from typing import Protocol, TypeAlias

from sqlalchemy.orm import Session

from app.agent.context import ToolExecutionContext
from app.agent.frameworks.langchain.runner import LangChainAgentResult
from app.agent.native_agent import NativeAgentResult
from app.agent.run_event import AgentRunEvent
from app.constants.agent_runtime import AgentRuntime


AgentExecutionResult: TypeAlias = NativeAgentResult | LangChainAgentResult


class AgentRuntimeExecutionService(Protocol):
    """Native / Framework 执行服务共同满足的同步与事件流接口。"""

    def run(
        self,
        *,
        db: Session,
        context: ToolExecutionContext,
        message: str,
    ) -> AgentExecutionResult:
        """执行一次同步 Agent Run。"""
        ...

    def run_events(
        self,
        *,
        db: Session,
        context: ToolExecutionContext,
        message: str,
    ) -> Iterator[AgentRunEvent]:
        """执行一次 provider-neutral Agent 事件流。"""
        ...


class AgentRuntimeUnavailableError(RuntimeError):
    """请求了当前部署未开放的 Agent Runtime。"""


class AgentRuntimeSelector:
    """
    为 HTTP 同步 / SSE 请求按需选择 Native 或 LangChain Candidate。

    使用 factory 而不是预先构造两个 Runtime，保证默认 Native 请求不会
    因为 Candidate 未启用而初始化 LangChain Model / Framework 依赖。
    """

    def __init__(
        self,
        *,
        native_factory: Callable[[], AgentRuntimeExecutionService],
        langchain_factory: Callable[[], AgentRuntimeExecutionService],
        langchain_candidate_enabled: bool,
    ) -> None:
        self._native_factory = native_factory
        self._langchain_factory = langchain_factory
        self._langchain_candidate_enabled = langchain_candidate_enabled

    def select(
        self,
        runtime: AgentRuntime,
    ) -> AgentRuntimeExecutionService:
        """返回请求对应的执行服务；Candidate 未开放时显式拒绝。"""

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
