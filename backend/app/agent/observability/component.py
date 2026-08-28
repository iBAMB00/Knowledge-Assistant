"""Provider-neutral component tracing for v2.5-A4."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.agent.observability.context import build_agent_span_context
from app.agent.observability.contracts import (
    AgentComponentCallContext,
    AgentComponentResult,
    AgentGraphExecutionMode,
    AgentObservationError,
    AgentObservationKind,
    AgentTraceContext,
)
from app.agent.observability.metrics import AgentRunMetricsCollector
from app.agent.observability.noop import NoOpComponentCallHandle
from app.agent.observability.provider import AgentComponentCallHandle, AgentTraceHandle

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _FailOpenComponentCallHandle:
    delegate: AgentComponentCallHandle
    kind: AgentObservationKind
    metrics_collector: AgentRunMetricsCollector | None = None
    started_ns: int | None = None
    _finished: bool = False

    @property
    def span_id(self) -> str:
        return self.delegate.span_id

    def finish(self, *, ok: bool = True, result: AgentComponentResult | None = None, error_code: str | None = None, error: AgentObservationError | None = None, warning: AgentObservationError | None = None) -> None:
        if self._finished:
            return
        try:
            if error is None and warning is None:
                self.delegate.finish(ok=ok, result=result, error_code=error_code)
            else:
                try:
                    self.delegate.finish(
                        ok=ok,
                        result=result,
                        error_code=error_code,
                        error=error,
                        warning=warning,
                    )
                except TypeError as exc:
                    if "unexpected keyword argument" not in str(exc):
                        raise
                    self.delegate.finish(
                        ok=ok,
                        result=result,
                        error_code=error_code,
                    )
        except Exception:
            logger.warning("Component observability finish failed; ignoring provider error", exc_info=True)
        finally:
            if self.metrics_collector is not None and self.started_ns is not None:
                self.metrics_collector.record_component(
                    kind=self.kind,
                    started_ns=self.started_ns,
                    ok=ok,
                    error=error,
                    warning=warning,
                )
            self._finished = True


@dataclass(frozen=True, slots=True)
class AgentComponentTracer:
    trace_context: AgentTraceContext
    trace_handle: AgentTraceHandle
    metrics_collector: AgentRunMetricsCollector | None = None

    def start_tool(self, *, tool_name: str, tool_version: str, tool_source: str, call_id: str, turn: int | None = None) -> AgentComponentCallHandle:
        return self._start(
            kind=AgentObservationKind.TOOL,
            name=f"tool.{tool_name}",
            turn=turn,
            call_id=call_id,
            tool_name=tool_name,
            tool_version=tool_version,
            tool_source=tool_source,
        )

    def start_retrieval(self, *, parent_span_id: str, top_k: int | None = None, turn: int | None = None) -> AgentComponentCallHandle:
        return self._start(
            kind=AgentObservationKind.RETRIEVAL,
            name="retrieval.knowledge",
            parent_span_id=parent_span_id,
            turn=turn,
            retrieval_mode="optimized",
            top_k=top_k,
        )

    def start_mcp(self, *, parent_span_id: str, tool_name: str, server_id: str, call_id: str, turn: int | None = None) -> AgentComponentCallHandle:
        return self._start(
            kind=AgentObservationKind.MCP,
            name=f"mcp.{tool_name}",
            parent_span_id=parent_span_id,
            turn=turn,
            call_id=call_id,
            tool_name=tool_name,
            mcp_server_id=server_id,
        )

    def start_graph_node(self, *, node_name: str, execution_mode: AgentGraphExecutionMode, turn: int | None = None) -> AgentComponentCallHandle:
        return self._start(
            kind=AgentObservationKind.GRAPH_NODE,
            name=f"graph.{node_name}",
            turn=turn,
            graph_node=node_name,
            graph_execution_mode=execution_mode,
        )

    def _start(self, *, kind: AgentObservationKind, name: str, parent_span_id: str | None = None, **metadata: object) -> AgentComponentCallHandle:
        span = build_agent_span_context(
            trace_context=self.trace_context,
            kind=kind,
            name=name,
            parent_span_id=parent_span_id,
        )
        call_context = AgentComponentCallContext(span=span, **metadata)
        started_ns = (
            self.metrics_collector.begin_call()
            if self.metrics_collector is not None
            else None
        )
        try:
            handle = self.trace_handle.start_component_call(call_context=call_context)
        except Exception:
            logger.warning("Component observability start failed; degrading to no-op", exc_info=True)
            handle = NoOpComponentCallHandle(span_id=span.span_id)
        return _FailOpenComponentCallHandle(
            handle,
            kind=kind,
            metrics_collector=self.metrics_collector,
            started_ns=started_ns,
        )
