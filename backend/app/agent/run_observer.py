from collections.abc import Sequence
from typing import Protocol

from app.agent.model_response import LLMToolCall
from app.agent.prompt_ops import AgentTracePromptLink


class AgentRunObserver(Protocol):
    """
    Native Agent 的可选进程内观察接口。

    用于 Eval / Trace 等需要读取运行事实的内部组件。它不是 SSE Contract，
    也不会自动持久化 Tool 参数或 Tool Result 正文。
    """

    def on_trace_started(self, link: AgentTracePromptLink) -> None:
        """Trace created for this run; contains identity/version metadata only."""
        ...

    def on_tool_call_requested(self, tool_call: LLMToolCall) -> None:
        """模型请求一次 Tool Call 时触发；此时尚未通过 Runtime 执行保护。"""
        ...

    def on_tool_result(
        self,
        *,
        call_id: str,
        tool_name: str,
        ok: bool,
        error_code: str | None,
        evidence_refs: Sequence[str],
    ) -> None:
        """一次 Tool Call 完成后触发；只附带无正文证据引用。"""
        ...

    def on_final_answer(self, answer: str) -> None:
        """最终回答产生时触发；Observer 自己决定是否仅提取安全元数据。"""
        ...


def notify_trace_started(
    observer: AgentRunObserver | None,
    link: AgentTracePromptLink,
) -> None:
    """Best-effort optional notification for legacy observers/test doubles."""

    if observer is None:
        return
    callback = getattr(observer, "on_trace_started", None)
    if callable(callback):
        callback(link)
