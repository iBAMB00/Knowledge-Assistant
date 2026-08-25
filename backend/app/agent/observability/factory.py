"""Observability provider composition with fail-open defaults."""

from __future__ import annotations

import logging
from collections.abc import Callable

from app.agent.observability.langfuse_provider import LangfuseObservabilityProvider
from app.agent.observability.noop import NoOpObservabilityProvider
from app.agent.observability.provider import ObservabilityProvider
from app.core.config import Settings

logger = logging.getLogger(__name__)


def build_observability_provider(
    settings: Settings,
    *,
    langfuse_builder: Callable[[Settings], ObservabilityProvider] | None = None,
) -> ObservabilityProvider:
    """Build the configured provider without making observability a hard dependency."""

    if not settings.langfuse_enabled:
        return NoOpObservabilityProvider()

    builder = langfuse_builder or LangfuseObservabilityProvider.from_settings
    try:
        return builder(settings)
    except Exception:
        logger.warning(
            "Langfuse initialization failed; Agent observability degraded to no-op",
            exc_info=True,
        )
        return NoOpObservabilityProvider()
