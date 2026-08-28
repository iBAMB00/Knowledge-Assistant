from app.agent.observability.contracts import AgentErrorStage
from app.agent.observability.error import build_observation_error


def test_structured_error_uses_cause_message_redacts_secrets_and_detects_http_status() -> None:
    try:
        try:
            raise RuntimeError(
                "reranker request failed: HTTP 403: token=super-secret-value"
            )
        except RuntimeError as exc:
            raise ValueError("generic wrapper") from exc
    except ValueError as exc:
        error = build_observation_error(
            exc,
            stage=AgentErrorStage.RETRIEVAL,
            error_code="execution_failed",
            provider="bailian",
            model="qwen3-rerank",
            retryable=False,
            fail_open=False,
        )

    assert error.error_code == "execution_failed"
    assert error.error_type == "ValueError"
    assert error.root_cause_type == "RuntimeError"
    assert error.stage is AgentErrorStage.RETRIEVAL
    assert error.substage == "reranker"
    assert error.http_status == 403
    assert error.provider == "bailian"
    assert error.model == "qwen3-rerank"
    assert error.safe_message is not None
    assert "super-secret-value" not in error.safe_message
    assert "<redacted>" in error.safe_message
    assert len(error.fingerprint) == 16


def test_safe_message_redacts_json_quoted_and_multiword_secrets() -> None:
    error = build_observation_error(
        RuntimeError(
            'HTTP 401: {"api_key":"supersecret","password":"two words",'
            '"authorization":"Bearer abc.def"}'
        ),
        stage=AgentErrorStage.MODEL,
        error_code="provider_failed",
    )

    message = error.safe_message or ""
    assert "supersecret" not in message
    assert "two words" not in message
    assert "abc.def" not in message
    assert message.count("<redacted>") >= 3
    assert error.http_status == 401


def test_structured_error_reads_typed_provider_diagnostics_from_cause_chain() -> None:
    from app.services.reranker.base import RerankerProviderError

    provider_error = RerankerProviderError(
        "reranker request failed: HTTP 403 (AllocationQuota.FreeTierOnly)",
        provider="bailian",
        model="qwen3-rerank",
        error_code="reranker_http_error",
        provider_error_code="AllocationQuota.FreeTierOnly",
        http_status=403,
        retryable=False,
        fail_open=False,
    )
    try:
        try:
            raise provider_error
        except RerankerProviderError as exc:
            raise RuntimeError("knowledge search failed") from exc
    except RuntimeError as exc:
        error = build_observation_error(
            exc,
            stage=AgentErrorStage.RETRIEVAL,
            error_code="execution_failed",
        )

    assert error.provider == "bailian"
    assert error.model == "qwen3-rerank"
    assert error.provider_error_code == "AllocationQuota.FreeTierOnly"
    assert error.http_status == 403
    assert error.retryable is False
    assert error.fail_open is False
    assert error.substage == "reranker"
