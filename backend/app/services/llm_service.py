import json
import logging
from collections.abc import Iterator, Sequence
from time import perf_counter
from typing import Any

from openai import OpenAI

from app.agent.agent_prompt import (
    AGENT_TOOL_CALLING_PROMPT_VERSION,
    build_agent_tool_calling_system_prompt,
    build_base_agent_system_prompt,
)
from app.agent.model_response import (
    LLMToolCall,
    LLMToolExchange,
    LLMToolResponse,
    LLMToolResult,
)
from app.agent.tool_result_message import build_model_facing_tool_result_content
from app.agent.tools.base import ToolContract
from app.core.config import get_settings


logger = logging.getLogger(__name__)


class LLMService:
    """
    LLM 模型能力服务。

    负责普通和流式模型调用，不记录完整 Prompt、
    企业知识正文或模型回答内容。
    """

    AGENT_PROMPT_VERSION = AGENT_TOOL_CALLING_PROMPT_VERSION

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

    def chat_with_tools(
        self,
        message: str,
        tool_contracts: Sequence[ToolContract],
    ) -> LLMToolResponse:
        """
        发起第一次 Tool Calling 模型调用。

        兼容 B1 入口；真正的 provider 调用统一由
        chat_with_tool_history() 完成。
        """

        return self.chat_with_tool_history(
            message=message,
            tool_contracts=tool_contracts,
            history=[],
        )

    def chat_with_tool_history(
        self,
        *,
        message: str,
        tool_contracts: Sequence[ToolContract],
        history: Sequence[LLMToolExchange],
    ) -> LLMToolResponse:
        """
        使用 provider-neutral Tool 历史继续一次 Tool Calling 对话。

        Agent Runtime 只保存 LLMToolExchange；
        OpenAI-compatible assistant/tool message 的序列化留在 LLMService。
        """

        normalized_message = self._normalize_message(message)
        tool_definitions = self._build_tool_definitions(tool_contracts)
        messages = self._build_tool_calling_messages(normalized_message)
        messages.extend(self._build_tool_history_messages(history))
        started_at = perf_counter()

        logger.info(
            "LLM call started: model=%s mode=tool_calling "
            "input_chars=%d tool_count=%d history_rounds=%d",
            self.model_name,
            len(normalized_message),
            len(tool_definitions),
            len(history),
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.2,
                tools=tool_definitions,
            )

            message_response = response.choices[0].message
            content = message_response.content
            tool_calls = self._extract_tool_calls(
                getattr(message_response, "tool_calls", None)
            )

            if not content and not tool_calls:
                raise RuntimeError(
                    "模型没有返回有效内容或 Tool Call"
                )

        except Exception as exc:
            logger.error(
                "LLM call failed: model=%s "
                "mode=tool_calling elapsed_ms=%d "
                "error_type=%s",
                self.model_name,
                self._elapsed_ms(started_at),
                type(exc).__name__,
            )
            raise

        logger.info(
            "LLM call completed: model=%s "
            "mode=tool_calling input_chars=%d "
            "output_chars=%d tool_call_count=%d "
            "history_rounds=%d elapsed_ms=%d",
            self.model_name,
            len(normalized_message),
            len(content or ""),
            len(tool_calls),
            len(history),
            self._elapsed_ms(started_at),
        )

        return LLMToolResponse(
            content=content,
            tool_calls=tool_calls,
        )

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

    @classmethod
    def _build_tool_calling_messages(
        cls,
        message: str,
    ) -> list[dict[str, str]]:
        """构建 Agent Tool Calling 消息，并冻结知识证据引用格式。"""

        messages = cls._build_messages(message)
        messages[0] = {
            "role": "system",
            "content": build_agent_tool_calling_system_prompt(),
        }
        return messages

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
                "content": build_base_agent_system_prompt(),
            },
            {
                "role": "user",
                "content": message,
            },
        ]

    @staticmethod
    def _build_tool_history_messages(
        history: Sequence[LLMToolExchange],
    ) -> list[dict[str, Any]]:
        """把内部 Tool Exchange 序列化为 provider message。"""

        messages: list[dict[str, Any]] = []

        for exchange in history:
            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": exchange.response.content,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.name,
                            "arguments": tool_call.arguments_json,
                        },
                    }
                    for tool_call in exchange.response.tool_calls
                ],
            }
            messages.append(assistant_message)

            for tool_result in exchange.tool_results:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_result.call_id,
                        "content": LLMService._build_tool_result_content(
                            tool_result
                        ),
                    }
                )

        return messages

    @staticmethod
    def _build_tool_result_content(tool_result: LLMToolResult) -> str:
        """委托共享序列化器构建模型可见 Tool Result。"""

        return build_model_facing_tool_result_content(tool_result)

    @staticmethod
    def _build_tool_definitions(
        tool_contracts: Sequence[ToolContract],
    ) -> list[dict[str, Any]]:
        """把内部 ToolContract 转换成 OpenAI-compatible tools schema。"""

        if not tool_contracts:
            raise ValueError("tool_contracts cannot be empty")

        names: set[str] = set()
        definitions: list[dict[str, Any]] = []

        for contract in tool_contracts:
            if contract.name in names:
                raise ValueError(
                    f"duplicate tool name: {contract.name}"
                )

            names.add(contract.name)
            definitions.append(
                {
                    "type": "function",
                    "function": {
                        "name": contract.name,
                        "description": contract.description,
                        "parameters": contract.input_schema,
                    },
                }
            )

        return definitions

    @staticmethod
    def _extract_tool_calls(
        provider_tool_calls: Any | None,
    ) -> list[LLMToolCall]:
        """把 provider Tool Call 适配为项目内部稳定结构。"""

        if not provider_tool_calls:
            return []

        return [
            LLMToolCall(
                id=tool_call.id,
                name=tool_call.function.name,
                arguments_json=tool_call.function.arguments,
            )
            for tool_call in provider_tool_calls
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
