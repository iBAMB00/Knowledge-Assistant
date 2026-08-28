"""Safe structured error helpers for Agent observability.

Only bounded, redacted diagnostic facts may cross the external observability
boundary. Full traceback/request/response payloads remain in application logs
and are correlated by request/trace/run identifiers.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from typing import Any

from app.agent.observability.contracts import AgentErrorStage, AgentObservationError

_MAX_SAFE_MESSAGE_CHARS = 320
_HTTP_STATUS_PATTERN = re.compile(r"\b(?:HTTP|status(?:_code)?)\s*[:= ]\s*([1-5][0-9]{2})\b", re.I)
_BEARER_PATTERN = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")
_BASIC_AUTH_PATTERN = re.compile(r"(?i)\bBasic\s+[A-Za-z0-9+/=]+")
_SECRET_VALUE_PATTERN = re.compile(
    r'''(?ix)
    (?P<prefix>
        ["']?(?:api[_-]?key|access[_-]?token|refresh[_-]?token|authorization|token|secret|password)["']?
        \s*[:=]\s*
    )
    (?P<value>
        "(?:\\.|[^"\\])*"
        |
        '(?:\\.|[^'\\])*'
        |
        [^\s,;}&]+)
    '''
)
_LANGFUSE_KEY_PATTERN = re.compile(r"\b(?:sk|pk)-lf-[A-Za-z0-9_-]+\b", re.I)
_GENERIC_SECRET_KEY_PATTERN = re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b", re.I)


def build_observation_error(
    exc: BaseException,
    *,
    stage: AgentErrorStage,
    error_code: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    retryable: bool | None = None,
    fail_open: bool | None = None,
) -> AgentObservationError:
    """Create a bounded vendor-safe diagnostic from one exception chain."""

    chain = tuple(_exception_chain(exc))
    root = chain[-1]
    safe_message = _select_safe_message(chain)

    resolved_provider = _normalize_token(provider) or _first_text_attr(chain, "provider")
    resolved_model = _normalize_token(model) or _first_text_attr(chain, "model")
    resolved_retryable = retryable if retryable is not None else _first_bool_attr(chain, "retryable")
    resolved_fail_open = fail_open if fail_open is not None else _first_bool_attr(chain, "fail_open")
    http_status = _extract_http_status(chain, safe_message)
    provider_error_code = _first_text_attr(chain, "provider_error_code")
    if provider_error_code is None:
        # Typed provider exceptions may only expose a stable internal error_code.
        provider_error_code = _typed_provider_error_code(chain)

    resolved_code = _normalize_token(error_code) or _typed_error_code(chain) or type(exc).__name__
    substage = _first_text_attr(chain, "substage") or _infer_substage(safe_message)

    fingerprint_material = "|".join(
        [
            stage.value,
            resolved_code,
            type(exc).__name__,
            type(root).__name__,
            substage or "",
            resolved_provider or "",
            resolved_model or "",
            provider_error_code or "",
            str(http_status or ""),
        ]
    )
    fingerprint = hashlib.sha256(
        fingerprint_material.encode("utf-8", errors="replace")
    ).hexdigest()[:16]

    return AgentObservationError(
        error_code=resolved_code,
        error_type=type(exc).__name__,
        stage=stage,
        substage=substage,
        safe_message=safe_message,
        root_cause_type=(type(root).__name__ if type(root) is not type(exc) else None),
        provider=resolved_provider,
        model=resolved_model,
        provider_error_code=provider_error_code,
        http_status=http_status,
        retryable=resolved_retryable,
        fail_open=resolved_fail_open,
        fingerprint=fingerprint,
    )


def build_control_error(
    *,
    error_code: str,
    stage: AgentErrorStage = AgentErrorStage.AGENT,
) -> AgentObservationError:
    """Represent a control-flow terminal state without inventing exception text."""

    normalized = _normalize_token(error_code) or "agent_run_failed"
    fingerprint = hashlib.sha256(
        f"{stage.value}|{normalized}".encode("utf-8")
    ).hexdigest()[:16]
    return AgentObservationError(
        error_code=normalized,
        error_type=normalized,
        stage=stage,
        fingerprint=fingerprint,
    )


def _exception_chain(exc: BaseException) -> Iterator[BaseException]:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _select_safe_message(chain: tuple[BaseException, ...]) -> str | None:
    # Prefer a meaningful root/provider cause over generic Tool/Runtime wrappers.
    candidates = list(chain[1:]) + list(chain[:1])
    for item in candidates:
        message = str(item).strip()
        if not message:
            continue
        sanitized = _sanitize_message(message)
        if sanitized:
            return sanitized
    return None


def _sanitize_message(message: str) -> str:
    sanitized = " ".join(message.split())
    sanitized = _BEARER_PATTERN.sub("Bearer <redacted>", sanitized)
    sanitized = _BASIC_AUTH_PATTERN.sub("Basic <redacted>", sanitized)
    sanitized = _SECRET_VALUE_PATTERN.sub(
        lambda match: f"{match.group('prefix')}<redacted>",
        sanitized,
    )
    sanitized = _LANGFUSE_KEY_PATTERN.sub("<redacted>", sanitized)
    sanitized = _GENERIC_SECRET_KEY_PATTERN.sub("<redacted>", sanitized)
    if len(sanitized) > _MAX_SAFE_MESSAGE_CHARS:
        sanitized = sanitized[: _MAX_SAFE_MESSAGE_CHARS - 1].rstrip() + "…"
    return sanitized


def _extract_http_status(
    chain: tuple[BaseException, ...],
    safe_message: str | None,
) -> int | None:
    for item in chain:
        for attr in ("http_status", "status_code", "code"):
            value = getattr(item, attr, None)
            if isinstance(value, bool):
                continue
            if isinstance(value, int) and 100 <= value <= 599:
                return value
    if safe_message:
        match = _HTTP_STATUS_PATTERN.search(safe_message)
        if match:
            return int(match.group(1))
    return None


def _typed_provider_error_code(chain: tuple[BaseException, ...]) -> str | None:
    for item in chain:
        if hasattr(item, "provider") and hasattr(item, "model"):
            value = getattr(item, "error_code", None)
            if isinstance(value, str) and value.strip():
                return value.strip()[:128]
    return None


def _typed_error_code(chain: tuple[BaseException, ...]) -> str | None:
    for item in chain:
        value = getattr(item, "error_code", None)
        if isinstance(value, str) and value.strip():
            return value.strip()[:128]
    return None


def _first_text_attr(chain: tuple[BaseException, ...], attr: str) -> str | None:
    for item in chain:
        value = getattr(item, attr, None)
        if isinstance(value, str):
            normalized = value.strip()
            if normalized:
                return normalized[:128]
    return None


def _first_bool_attr(chain: tuple[BaseException, ...], attr: str) -> bool | None:
    for item in chain:
        value = getattr(item, attr, None)
        if isinstance(value, bool):
            return value
    return None


def _infer_substage(message: str | None) -> str | None:
    if not message:
        return None
    lowered = message.lower()
    if "rerank" in lowered:
        return "reranker"
    if "embedding" in lowered or "embed" in lowered:
        return "embedding"
    if "qdrant" in lowered or "vector" in lowered:
        return "vector_store"
    if "mcp" in lowered:
        return "mcp_transport"
    if "timeout" in lowered or "timed out" in lowered:
        return "timeout"
    return None


def _normalize_token(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None
