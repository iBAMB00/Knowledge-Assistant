import logging
from types import SimpleNamespace
from typing import Any

import pytest

import app.services.llm_service as llm_service_module
from app.services.llm_service import LLMService


class FakeStream:
    """
    测试使用的模型流。
    """

    def __init__(
        self,
        chunks: list[Any],
    ) -> None:
        self.chunks = chunks
        self.closed = False

    def __iter__(self):
        return iter(self.chunks)

    def close(self) -> None:
        self.closed = True


class FakeCompletions:
    """
    测试使用的模型 completions 客户端。
    """

    def __init__(
        self,
        stream: FakeStream,
        answer: str = "测试回答",
    ) -> None:
        self.stream = stream
        self.answer = answer
        self.calls: list[dict[str, Any]] = []

    def create(
        self,
        **kwargs: Any,
    ) -> Any:
        self.calls.append(kwargs)

        if kwargs.get("stream"):
            return self.stream

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=self.answer,
                    )
                )
            ]
        )


class FakeOpenAIClient:
    """
    测试使用的 OpenAI 客户端。
    """

    def __init__(
        self,
        completions: FakeCompletions,
    ) -> None:
        self.chat = SimpleNamespace(
            completions=completions
        )


def build_service(
    monkeypatch: pytest.MonkeyPatch,
    completions: FakeCompletions,
) -> LLMService:
    """
    创建不调用真实模型的 LLMService。
    """

    settings = SimpleNamespace(
        model_name="test-model",
        model_api_key="test-key",
        model_base_url="http://test-model",
    )

    monkeypatch.setattr(
        llm_service_module,
        "get_settings",
        lambda: settings,
    )

    monkeypatch.setattr(
        llm_service_module,
        "OpenAI",
        lambda **_: FakeOpenAIClient(
            completions
        ),
    )

    return LLMService()


def build_chunk(
    content: str | None,
) -> Any:
    """
    创建模型流式返回片段。
    """

    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    content=content,
                )
            )
        ]
    )


def test_stream_chat_returns_model_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    验证流式调用逐块返回非空内容。
    """

    stream = FakeStream(
        chunks=[
            SimpleNamespace(choices=[]),
            build_chunk(None),
            build_chunk("管理员"),
            build_chunk("可以重置密码。"),
        ]
    )

    completions = FakeCompletions(
        stream=stream
    )

    service = build_service(
        monkeypatch=monkeypatch,
        completions=completions,
    )

    contents = list(
        service.stream_chat(
            "如何重置密码？"
        )
    )

    assert contents == [
        "管理员",
        "可以重置密码。",
    ]

    assert completions.calls[0][
        "stream"
    ] is True

    assert stream.closed is True


def test_llm_logs_do_not_contain_prompt_or_answer(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    验证日志不记录完整 Prompt 和回答。
    """

    secret_prompt = (
        "企业机密：管理员初始密码为 ABC123。"
    )
    secret_answer = (
        "企业机密回答：请勿向外部人员公开。"
    )

    stream = FakeStream(chunks=[])

    completions = FakeCompletions(
        stream=stream,
        answer=secret_answer,
    )

    service = build_service(
        monkeypatch=monkeypatch,
        completions=completions,
    )

    caplog.set_level(
        logging.INFO,
        logger="app.services.llm_service",
    )

    answer = service.chat(secret_prompt)

    assert answer == secret_answer

    assert secret_prompt not in caplog.text
    assert secret_answer not in caplog.text

    assert "input_chars=" in caplog.text
    assert "output_chars=" in caplog.text
    assert "elapsed_ms=" in caplog.text


def test_stream_chat_closes_stream_when_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    验证调用方停止读取时关闭底层模型流。
    """

    stream = FakeStream(
        chunks=[
            build_chunk("第一段"),
            build_chunk("第二段"),
        ]
    )

    completions = FakeCompletions(
        stream=stream
    )

    service = build_service(
        monkeypatch=monkeypatch,
        completions=completions,
    )

    iterator = service.stream_chat(
        "测试中断"
    )

    assert next(iterator) == "第一段"

    iterator.close()

    assert stream.closed is True


@pytest.mark.parametrize(
    "message",
    [
        "",
        "   ",
    ],
)
def test_stream_chat_rejects_empty_message(
    monkeypatch: pytest.MonkeyPatch,
    message: str,
) -> None:
    """
    验证流式调用拒绝空消息。
    """

    stream = FakeStream(chunks=[])

    completions = FakeCompletions(
        stream=stream
    )

    service = build_service(
        monkeypatch=monkeypatch,
        completions=completions,
    )

    with pytest.raises(
        ValueError,
        match="message cannot be empty",
    ):
        list(service.stream_chat(message))