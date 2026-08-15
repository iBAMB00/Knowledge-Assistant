import json
from dataclasses import dataclass
from typing import Any

from app.agent.model_response import LLMToolCall
from app.schemas.agent_evaluation import AgentObservedToolCall


@dataclass
class _ObservedCall:
    call_id: str
    tool_name: str
    arguments: dict[str, Any]
    error_code: str | None = None


class AgentEvaluationObservationCollector:
    """
    D2 进程内 Agent Observer。

    只在 Eval 运行期间捕获模型请求的 Tool 名称与参数，并在 Tool 执行完成后
    追加安全 error_code。数据不会自动进入 SSE、AgentRun 或 ToolCall 数据库。
    """

    def __init__(self) -> None:
        self._calls: list[_ObservedCall] = []
        self._call_indexes: dict[str, int] = {}

    def on_tool_call_requested(self, tool_call: LLMToolCall) -> None:
        arguments = self._parse_arguments(tool_call.arguments_json)
        self._calls.append(
            _ObservedCall(
                call_id=tool_call.id,
                tool_name=tool_call.name,
                arguments=arguments,
            )
        )
        self._call_indexes[tool_call.id] = len(self._calls) - 1

    def on_tool_result(
        self,
        *,
        call_id: str,
        tool_name: str,
        ok: bool,
        error_code: str | None,
    ) -> None:
        index = self._call_indexes.get(call_id)
        if index is None:
            return

        observed = self._calls[index]
        if observed.tool_name != tool_name:
            return

        observed.error_code = None if ok else (error_code or "tool_error")

    def build_tool_calls(self) -> list[AgentObservedToolCall]:
        return [
            AgentObservedToolCall(
                tool_name=observed.tool_name,
                arguments=observed.arguments,
                error_code=observed.error_code,
            )
            for observed in self._calls
        ]

    @staticmethod
    def _parse_arguments(arguments_json: str) -> dict[str, Any]:
        try:
            parsed = json.loads(arguments_json)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
