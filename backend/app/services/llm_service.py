import logging
from collections.abc import Iterator
from time import perf_counter
from typing import Any

from openai import OpenAI

from app.core.config import get_settings


logger = logging.getLogger(__name__)


class LLMService:
    """
    LLM 模型能力服务。

    负责普通和流式模型调用，不记录完整 Prompt、
    企业知识正文或模型回答内容。
    """

    def __init__(self) -> None:
        settings = get_settings()

        self.model_name = settings.model_name
        self.client = OpenAI(
            api_key=settings.model_api_key,
            base_url=settings.model_base_url,
            timeout=60,
        )

    def chat(self, message: str) -> str:
        """
        调用模型并返回完整回答。
        """

        normalized_message = self._normalize_message(message)
        started_at = perf_counter()

        logger.info(
            "LLM call started: model=%s mode=non_stream "
            "input_chars=%d",
            self.model_name,
            len(normalized_message),
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=self._build_messages(
                    normalized_message
                ),
                temperature=0.2,
            )

            content = response.choices[0].message.content

            if not content:
                raise RuntimeError(
                    "模型没有返回有效内容"
                )

        except Exception as exc:
            logger.error(
                "LLM call failed: model=%s "
                "mode=non_stream elapsed_ms=%d "
                "error_type=%s",
                self.model_name,
                self._elapsed_ms(started_at),
                type(exc).__name__,
            )
            raise

        logger.info(
            "LLM call completed: model=%s "
            "mode=non_stream input_chars=%d "
            "output_chars=%d elapsed_ms=%d",
            self.model_name,
            len(normalized_message),
            len(content),
            self._elapsed_ms(started_at),
        )

        return content

    def stream_chat(
        self,
        message: str,
    ) -> Iterator[str]:
        """
        调用模型并逐块返回回答内容。

        流结束、异常或调用方主动关闭生成器时，
        都会尝试关闭底层模型流。
        """

        normalized_message = self._normalize_message(message)
        started_at = perf_counter()

        stream: Any | None = None
        output_chars = 0
        chunk_count = 0

        logger.info(
            "LLM call started: model=%s mode=stream "
            "input_chars=%d",
            self.model_name,
            len(normalized_message),
        )

        try:
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=self._build_messages(
                    normalized_message
                ),
                temperature=0.2,
                stream=True,
            )

            for chunk in stream:
                if not chunk.choices:
                    continue

                content = (
                    chunk.choices[0].delta.content
                )

                if not content:
                    continue

                chunk_count += 1
                output_chars += len(content)

                yield content

        except GeneratorExit:
            logger.info(
                "LLM stream cancelled: model=%s "
                "chunks=%d output_chars=%d "
                "elapsed_ms=%d",
                self.model_name,
                chunk_count,
                output_chars,
                self._elapsed_ms(started_at),
            )
            raise

        except Exception as exc:
            logger.error(
                "LLM call failed: model=%s "
                "mode=stream chunks=%d "
                "output_chars=%d elapsed_ms=%d "
                "error_type=%s",
                self.model_name,
                chunk_count,
                output_chars,
                self._elapsed_ms(started_at),
                type(exc).__name__,
            )
            raise

        else:
            logger.info(
                "LLM call completed: model=%s "
                "mode=stream chunks=%d "
                "output_chars=%d elapsed_ms=%d",
                self.model_name,
                chunk_count,
                output_chars,
                self._elapsed_ms(started_at),
            )

        finally:
            self._close_stream(stream)

    @staticmethod
    def _normalize_message(message: str) -> str:
        """
        校验并标准化模型输入。
        """

        if not message or not message.strip():
            raise ValueError(
                "message cannot be empty"
            )

        return message.strip()

    @staticmethod
    def _build_messages(
        message: str,
    ) -> list[dict[str, str]]:
        """
        构建模型消息。
        """

        return [
            {
                "role": "system",
                "content": (
                    "你是一个企业私有知识助手，"
                    "请基于已知信息准确、简洁地"
                    "回答用户问题。"
                ),
            },
            {
                "role": "user",
                "content": message,
            },
        ]

    @staticmethod
    def _elapsed_ms(started_at: float) -> int:
        """
        计算调用耗时，单位为毫秒。
        """

        return round(
            (perf_counter() - started_at) * 1000
        )

    @staticmethod
    def _close_stream(
        stream: Any | None,
    ) -> None:
        """
        尝试关闭底层模型流。

        关闭失败只记录异常类型，不覆盖原始业务异常。
        """

        if stream is None:
            return

        close_method = getattr(
            stream,
            "close",
            None,
        )

        if not callable(close_method):
            return

        try:
            close_method()

        except Exception as exc:
            logger.warning(
                "Failed to close LLM stream: "
                "error_type=%s",
                type(exc).__name__,
            )