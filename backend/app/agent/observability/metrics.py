"""Run-level metrics aggregation for v2.5-A5 AgentOps."""

from __future__ import annotations

import time
from decimal import Decimal
from threading import Lock
from typing import Callable

from app.agent.observability.contracts import (
    AgentModelUsage,
    AgentObservationKind,
    AgentRunMetrics,
)
from app.agent.observability.pricing import AgentModelPricing, estimate_model_cost_usd


class AgentRunMetricsCollector:
    """Thread-safe in-process aggregation independent of the trace vendor."""

    def __init__(
        self,
        *,
        pricing: AgentModelPricing | None = None,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self._pricing = pricing
        self._clock_ns = clock_ns
        self._run_started_ns = clock_ns()
        self._lock = Lock()
        self._snapshot: AgentRunMetrics | None = None

        self._model_calls = 0
        self._tool_calls = 0
        self._retrieval_calls = 0
        self._mcp_calls = 0
        self._graph_node_calls = 0
        self._failed_calls = 0

        self._input_tokens = 0
        self._output_tokens = 0
        self._total_tokens = 0
        self._model_usage_missing_calls = 0
        self._costed_model_calls = 0
        self._estimated_cost_usd = Decimal("0")

        self._model_latency_ms = 0.0
        self._tool_latency_ms = 0.0
        self._retrieval_latency_ms = 0.0
        self._mcp_latency_ms = 0.0
        self._graph_node_latency_ms = 0.0

    @property
    def pricing(self) -> AgentModelPricing | None:
        return self._pricing

    def begin_call(self) -> int:
        return self._clock_ns()

    def record_model(
        self,
        *,
        started_ns: int,
        ok: bool,
        usage: AgentModelUsage | None,
    ) -> None:
        elapsed_ms = self._elapsed_ms(started_ns)
        with self._lock:
            if self._snapshot is not None:
                return
            self._model_calls += 1
            self._model_latency_ms += elapsed_ms
            if not ok:
                self._failed_calls += 1

            if usage is None:
                self._model_usage_missing_calls += 1
                return

            if usage.input_tokens is not None:
                self._input_tokens += usage.input_tokens
            if usage.output_tokens is not None:
                self._output_tokens += usage.output_tokens
            if usage.total_tokens is not None:
                self._total_tokens += usage.total_tokens
            elif usage.input_tokens is not None and usage.output_tokens is not None:
                self._total_tokens += usage.input_tokens + usage.output_tokens

            if usage.input_tokens is None or usage.output_tokens is None:
                self._model_usage_missing_calls += 1
                return

            if self._pricing is not None:
                cost = estimate_model_cost_usd(pricing=self._pricing, usage=usage)
                if cost is not None:
                    self._estimated_cost_usd += cost
                    self._costed_model_calls += 1

    def record_component(
        self,
        *,
        kind: AgentObservationKind,
        started_ns: int,
        ok: bool,
    ) -> None:
        elapsed_ms = self._elapsed_ms(started_ns)
        with self._lock:
            if self._snapshot is not None:
                return
            if kind is AgentObservationKind.TOOL:
                self._tool_calls += 1
                self._tool_latency_ms += elapsed_ms
            elif kind is AgentObservationKind.RETRIEVAL:
                self._retrieval_calls += 1
                self._retrieval_latency_ms += elapsed_ms
            elif kind is AgentObservationKind.MCP:
                self._mcp_calls += 1
                self._mcp_latency_ms += elapsed_ms
            elif kind is AgentObservationKind.GRAPH_NODE:
                self._graph_node_calls += 1
                self._graph_node_latency_ms += elapsed_ms
            if not ok:
                self._failed_calls += 1

    def finish(
        self,
        *,
        success: bool,
        error_type: str | None = None,
    ) -> AgentRunMetrics:
        with self._lock:
            if self._snapshot is not None:
                return self._snapshot

            pricing_configured = self._pricing is not None
            cost_estimate_complete = (
                pricing_configured
                and self._model_usage_missing_calls == 0
                and self._costed_model_calls == self._model_calls
            )
            has_cost_evidence = self._model_calls == 0 or self._costed_model_calls > 0
            estimated_cost = (
                self._estimated_cost_usd.quantize(Decimal("0.00000001"))
                if pricing_configured and has_cost_evidence
                else None
            )
            pricing_version = self._pricing.version if self._pricing is not None else None

            self._snapshot = AgentRunMetrics(
                success=success,
                error_type=error_type,
                run_latency_ms=self._elapsed_ms(self._run_started_ns),
                model_calls=self._model_calls,
                tool_calls=self._tool_calls,
                retrieval_calls=self._retrieval_calls,
                mcp_calls=self._mcp_calls,
                graph_node_calls=self._graph_node_calls,
                failed_calls=self._failed_calls,
                input_tokens=self._input_tokens,
                output_tokens=self._output_tokens,
                total_tokens=self._total_tokens,
                model_usage_missing_calls=self._model_usage_missing_calls,
                model_latency_ms=self._round_ms(self._model_latency_ms),
                tool_latency_ms=self._round_ms(self._tool_latency_ms),
                retrieval_latency_ms=self._round_ms(self._retrieval_latency_ms),
                mcp_latency_ms=self._round_ms(self._mcp_latency_ms),
                graph_node_latency_ms=self._round_ms(self._graph_node_latency_ms),
                estimated_cost_usd=estimated_cost,
                pricing_configured=pricing_configured,
                cost_estimate_complete=cost_estimate_complete,
                pricing_version=pricing_version,
            )
            return self._snapshot

    def _elapsed_ms(self, started_ns: int) -> float:
        return self._round_ms(max(0, self._clock_ns() - started_ns) / 1_000_000)

    @staticmethod
    def _round_ms(value: float) -> float:
        return round(value, 3)
