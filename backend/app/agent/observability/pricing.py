"""Provider-neutral model pricing used for v2.5-A5 cost estimation."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from app.agent.observability.contracts import AgentModelCost, AgentModelUsage
from app.core.config import Settings

_ONE_MILLION = Decimal("1000000")


@dataclass(frozen=True, slots=True)
class AgentModelPricing:
    """Explicit pricing snapshot for the model used by one Agent runtime.

    Pricing is configuration, not a model/provider fact hard-coded into runtime
    code. This keeps historical cost estimates explainable when vendor prices
    change later.
    """

    provider: str
    model_name: str
    input_usd_per_million_tokens: Decimal
    output_usd_per_million_tokens: Decimal
    version: str


def build_agent_model_pricing(settings: Settings) -> AgentModelPricing | None:
    """Build pricing only when both input and output prices are configured."""

    input_price = settings.agent_model_input_cost_per_million_tokens_usd
    output_price = settings.agent_model_output_cost_per_million_tokens_usd
    if input_price is None or output_price is None:
        return None

    return AgentModelPricing(
        provider=settings.model_provider.strip(),
        model_name=settings.model_name.strip(),
        input_usd_per_million_tokens=input_price,
        output_usd_per_million_tokens=output_price,
        version=settings.agent_model_pricing_version.strip(),
    )


def estimate_model_cost_details_usd(
    *,
    pricing: AgentModelPricing,
    usage: AgentModelUsage,
) -> AgentModelCost | None:
    """Estimate one generation's explicit USD input/output/total buckets."""

    if usage.input_tokens is None or usage.output_tokens is None:
        return None

    input_cost = (
        Decimal(usage.input_tokens)
        * pricing.input_usd_per_million_tokens
        / _ONE_MILLION
    )
    output_cost = (
        Decimal(usage.output_tokens)
        * pricing.output_usd_per_million_tokens
        / _ONE_MILLION
    )
    total_cost = input_cost + output_cost
    return AgentModelCost(
        input_cost_usd=input_cost,
        output_cost_usd=output_cost,
        total_cost_usd=total_cost,
    )


def estimate_model_cost_usd(
    *,
    pricing: AgentModelPricing,
    usage: AgentModelUsage,
) -> Decimal | None:
    """Estimate one generation total cost from explicit token usage."""

    details = estimate_model_cost_details_usd(pricing=pricing, usage=usage)
    return details.total_cost_usd if details is not None else None
