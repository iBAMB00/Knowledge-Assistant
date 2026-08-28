"""LangChain model-call middleware bridge to provider-neutral Agent observability."""

from __future__ import annotations

from typing import Any

from app.agent.observability.contracts import AgentErrorStage
from app.agent.observability.error import build_observation_error
from app.agent.observability.model import AgentModelTracer, extract_langchain_usage


class LangChainModelObservabilityBridge:
    """Trace actual LangChain model handler calls without reading prompt/output bodies."""

    BRIDGE_VERSION = "1.0.0"

    def __init__(self, tracer: AgentModelTracer) -> None:
        self._tracer = tracer
        self._turn = 0

    def build_middleware(self) -> Any:
        AgentMiddleware = self._load_agent_middleware()
        bridge = self

        class ModelObservabilityMiddleware(AgentMiddleware):
            def wrap_model_call(self, request, handler):  # noqa: ANN001
                return bridge.observe_model_call(request=request, handler=handler)

        return ModelObservabilityMiddleware()

    def observe_model_call(self, *, request: Any, handler: Any) -> Any:
        # A3 deliberately does not inspect request prompt/messages.
        self._turn += 1
        handle = self._tracer.start_call(turn=self._turn)
        try:
            response = handler(request)
        except Exception as exc:
            handle.finish(
                ok=False,
                error_code=type(exc).__name__,
                error=build_observation_error(
                    exc,
                    stage=AgentErrorStage.MODEL,
                    error_code=type(exc).__name__,
                    provider=self._tracer.model_provider,
                    model=self._tracer.model_name,
                ),
            )
            raise

        handle.finish(
            ok=True,
            usage=extract_langchain_usage(response),
        )
        return response

    @staticmethod
    def _load_agent_middleware():
        try:
            from langchain.agents.middleware import AgentMiddleware
        except ImportError as exc:
            raise RuntimeError(
                "LangChain model observability requires langchain; "
                "install project requirements before using framework integration"
            ) from exc
        return AgentMiddleware
