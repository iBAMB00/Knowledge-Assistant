"""LangChain 实际 Tool execution 的框架中立观察接口。"""

from typing import Protocol


class LangChainToolExecutionObserver(Protocol):
    """
    观察真正进入 ToolNode 执行阶段的 Tool Call。

    与 AgentRunObserver 不同：
    - AgentRunObserver 观察模型“请求了什么”，用于 Eval / Trace；
    - 本接口只观察已经通过 Runtime Guard、实际开始执行的 Tool，
      用于 AgentToolCall 生命周期持久化。
    """

    def on_tool_execution_started(
        self,
        *,
        call_id: str,
        tool_name: str,
    ) -> None:
        """Tool 真正开始执行前触发。"""
        ...

    def on_tool_execution_finished(
        self,
        *,
        call_id: str,
        tool_name: str,
        ok: bool,
        error_code: str | None,
        duration_ms: int,
    ) -> None:
        """Tool 执行完成或异常退出后触发，只携带安全元数据。"""
        ...
