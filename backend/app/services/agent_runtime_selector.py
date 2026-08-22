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
        ...

    def run_events(
        self,
        *,
        db: Session,
        context: ToolExecutionContext,
        message: str,
    ) -> Iterator[AgentRunEvent]:
        ...


class AgentRuntimeUnavailableError(RuntimeError):
    """请求了当前部署未开放的 Agent Runtime。"""


class AgentRuntimeSelector:
    """按显式 feature gate 懒加载 Native / LangChain / LangGraph Runtime。"""

    def __init__(
        self,
        *,
        native_factory: Callable[[], AgentRuntimeExecutionService],
        langchain_factory: Callable[[], AgentRuntimeExecutionService],
        langchain_candidate_enabled: bool,
        langgraph_factory: Callable[[], AgentRuntimeExecutionService] | None = None,
        langgraph_candidate_enabled: bool = False,
    ) -> None:
        self._native_factory = native_factory
        self._langchain_factory = langchain_factory
        self._langchain_candidate_enabled = langchain_candidate_enabled
        self._langgraph_factory = langgraph_factory
        self._langgraph_candidate_enabled = langgraph_candidate_enabled

    def ensure_available(self, runtime: AgentRuntime) -> None:
        """仅检查部署开关，不触发模型/Tool/Graph 构造。"""

        if runtime is AgentRuntime.NATIVE:
            return
        if runtime is AgentRuntime.LANGCHAIN:
            if self._langchain_candidate_enabled:
                return
            raise AgentRuntimeUnavailableError(
                "langchain candidate runtime is disabled"
            )
        if runtime is AgentRuntime.LANGGRAPH:
            if (
                self._langgraph_candidate_enabled
                and self._langgraph_factory is not None
            ):
                return
            raise AgentRuntimeUnavailableError(
                "langgraph candidate runtime is disabled"
            )
        raise AgentRuntimeUnavailableError(
            f"unsupported agent runtime: {runtime}"
        )

    def select(
        self,
        runtime: AgentRuntime,
    ) -> AgentRuntimeExecutionService:
        """返回请求对应的执行服务；未开放 Candidate 不调用 factory。"""

        self.ensure_available(runtime)

        if runtime is AgentRuntime.NATIVE:
            return self._native_factory()
        if runtime is AgentRuntime.LANGCHAIN:
            return self._langchain_factory()
        if runtime is AgentRuntime.LANGGRAPH:
            if self._langgraph_factory is None:
                raise AgentRuntimeUnavailableError(
                    "langgraph candidate runtime is unavailable"
                )
            return self._langgraph_factory()

        raise AgentRuntimeUnavailableError(
            f"unsupported agent runtime: {runtime}"
        )
