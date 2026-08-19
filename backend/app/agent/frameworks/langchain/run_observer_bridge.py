"""LangChain Agent Middleware 到项目 AgentRunObserver 的观察桥接。"""

import json
import time
from collections.abc import Mapping, Sequence
from typing import Any

from app.agent.frameworks.langchain.execution_observer import (
    LangChainToolExecutionObserver,
)
from app.agent.model_response import LLMToolCall
from app.agent.run_observer import AgentRunObserver


class LangChainRunObserverBridge:
    """
    把 LangChain create_agent 生命周期事件归一化为现有 AgentRunObserver。

    设计目标：
    - Eval / Trace 继续消费项目自己的 provider-neutral Observer Contract；
    - 模型 Tool Call 在执行前被观察，Tool Result 只暴露安全 error_code / source_ref；
    - 不把 Tool Result 正文或隐藏推理写入 Observation。
    """

    BRIDGE_VERSION = "1.1.0"

    def __init__(
        self,
        observer: AgentRunObserver | None = None,
        *,
        execution_observer: LangChainToolExecutionObserver | None = None,
    ) -> None:
        if observer is None and execution_observer is None:
            raise ValueError("at least one observer is required")
        self._observer = observer
        self._execution_observer = execution_observer

    def build_middleware(self) -> Any:
        """构建一次请求级 LangChain Middleware；LangChain 依赖延迟加载。"""

        AgentMiddleware = self._load_agent_middleware()
        bridge = self

        class ObserverMiddleware(AgentMiddleware):
            """仅负责把 LangChain 生命周期事实转发给项目 Observer。"""

            def after_model(self, state, runtime):  # noqa: ANN001
                bridge.observe_model_state(state)
                return None

            def wrap_tool_call(self, request, handler):  # noqa: ANN001
                return bridge.observe_tool_execution(
                    request=request,
                    handler=handler,
                )

        return ObserverMiddleware()

    def observe_model_state(self, state: Mapping[str, Any]) -> None:
        """观察最新模型响应中的 Tool Call 请求。"""

        messages = state.get("messages")
        if not isinstance(messages, (list, tuple)) or not messages:
            return

        if self._observer is None:
            return

        message = messages[-1]
        for raw_call in self._extract_tool_calls(message):
            tool_call = self._normalize_tool_call(raw_call)
            if tool_call is not None:
                self._observer.on_tool_call_requested(tool_call)

    def observe_tool_execution(self, *, request: Any, handler: Any) -> Any:
        """围绕一次 LangChain Tool 执行，转发安全 Tool Result 元数据。"""

        raw_call = getattr(request, "tool_call", None)
        if not isinstance(raw_call, Mapping):
            return handler(request)

        call_id = self._normalize_text(raw_call.get("id"))
        tool_name = self._normalize_text(raw_call.get("name"))
        if not call_id or not tool_name:
            return handler(request)

        if self._execution_observer is not None:
            self._execution_observer.on_tool_execution_started(
                call_id=call_id,
                tool_name=tool_name,
            )

        started_at = time.perf_counter()
        try:
            result = handler(request)
        except Exception:
            duration_ms = max(
                0,
                int((time.perf_counter() - started_at) * 1000),
            )
            if self._observer is not None:
                self._observer.on_tool_result(
                    call_id=call_id,
                    tool_name=tool_name,
                    ok=False,
                    error_code="framework_tool_error",
                    evidence_refs=[],
                )
            if self._execution_observer is not None:
                self._execution_observer.on_tool_execution_finished(
                    call_id=call_id,
                    tool_name=tool_name,
                    ok=False,
                    error_code="framework_tool_error",
                    duration_ms=duration_ms,
                )
            raise

        duration_ms = max(
            0,
            int((time.perf_counter() - started_at) * 1000),
        )
        ok, error_code, evidence_refs = self._parse_tool_result(result)
        if self._observer is not None:
            self._observer.on_tool_result(
                call_id=call_id,
                tool_name=tool_name,
                ok=ok,
                error_code=error_code,
                evidence_refs=evidence_refs,
            )
        if self._execution_observer is not None:
            self._execution_observer.on_tool_execution_finished(
                call_id=call_id,
                tool_name=tool_name,
                ok=ok,
                error_code=error_code,
                duration_ms=duration_ms,
            )
        return result

    @staticmethod
    def _extract_tool_calls(message: Any) -> list[Any]:
        if isinstance(message, Mapping):
            value = message.get("tool_calls")
        else:
            value = getattr(message, "tool_calls", None)

        return list(value) if isinstance(value, (list, tuple)) else []

    @classmethod
    def _normalize_tool_call(cls, raw_call: Any) -> LLMToolCall | None:
        if not isinstance(raw_call, Mapping):
            return None

        call_id = cls._normalize_text(raw_call.get("id"))
        name = cls._normalize_text(raw_call.get("name"))
        arguments = raw_call.get("args", raw_call.get("arguments", {}))
        if not call_id or not name:
            return None

        if isinstance(arguments, str):
            arguments_json = arguments.strip() or "{}"
        else:
            try:
                arguments_json = json.dumps(
                    arguments if arguments is not None else {},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            except (TypeError, ValueError):
                arguments_json = "{}"

        return LLMToolCall(
            id=call_id,
            name=name,
            arguments_json=arguments_json,
        )

    @classmethod
    def _parse_tool_result(
        cls,
        result: Any,
    ) -> tuple[bool, str | None, list[str]]:
        content = cls._extract_result_content(result)
        if not isinstance(content, str):
            return True, None, []

        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            return True, None, []

        if not isinstance(payload, dict):
            return True, None, []

        if payload.get("ok") is False:
            error = payload.get("error")
            error_code = (
                cls._normalize_text(error.get("code"))
                if isinstance(error, Mapping)
                else None
            )
            return False, error_code or "tool_error", []

        evidence_refs = cls._normalize_refs(
            payload.get("evidence_refs")
            or payload.get("_available_source_refs")
            or []
        )
        return True, None, evidence_refs

    @classmethod
    def _extract_result_content(cls, result: Any) -> Any:
        if isinstance(result, Mapping):
            if "content" in result:
                return result.get("content")
            return cls._extract_content_from_update(result.get("update"))

        content = getattr(result, "content", None)
        if content is not None:
            return content

        return cls._extract_content_from_update(
            getattr(result, "update", None)
        )

    @staticmethod
    def _extract_content_from_update(update: Any) -> Any:
        if not isinstance(update, Mapping):
            return None
        messages = update.get("messages")
        if not isinstance(messages, (list, tuple)) or not messages:
            return None
        last = messages[-1]
        if isinstance(last, Mapping):
            return last.get("content")
        return getattr(last, "content", None)

    @classmethod
    def _normalize_refs(cls, values: Any) -> list[str]:
        if not isinstance(values, (list, tuple)):
            return []

        refs: list[str] = []
        seen: set[str] = set()
        for value in values:
            normalized = cls._normalize_text(value)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            refs.append(normalized)
        return refs

    @staticmethod
    def _normalize_text(value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        normalized = value.strip()
        return normalized or None

    @staticmethod
    def _load_agent_middleware():
        try:
            from langchain.agents.middleware import AgentMiddleware
        except ImportError as exc:
            raise RuntimeError(
                "LangChain Observer Bridge requires langchain; "
                "install project requirements before using v2.1 framework integration"
            ) from exc
        return AgentMiddleware
