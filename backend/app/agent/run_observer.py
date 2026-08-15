from typing import Protocol

from app.agent.model_response import LLMToolCall


class AgentRunObserver(Protocol):
    """
    Native Agent 的可选进程内观察接口。

    用于 Eval / Trace 等需要读取运行事实的内部组件。它不是 SSE Contract，
    也不会自动持久化 Tool 参数或 Tool Result 正文。
    """

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
    ) -> None:
        """一次真正执行的 Tool Call 得到安全结果摘要后触发。"""
        ...
