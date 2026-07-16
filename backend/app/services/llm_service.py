from openai import OpenAI

from app.core.config import get_settings
from collections.abc import Iterable

import logging

logger = logging.getLogger(__name__)

'''
LLM模型能力服务
负责模型调用，返回模型回复
'''

class LLMService:
    def __init__(self) -> None:
        settings = get_settings()

        self.model_name = settings.model_name
        self.client = OpenAI(
            api_key=settings.model_api_key,
            base_url=settings.model_base_url,
            timeout=60,
        )


    def chat(self, message: str) -> str:

        if not message or not message.strip():
            raise ValueError("message cannot be empty")

        logger.info(f"正在调用模型：{self.model_name}，用户问题：{message}")

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=self._build_messages(message),
            temperature=0.2,
        )

        content = response.choices[0].message.content

        if not content:
            raise RuntimeError("模型没有返回有效内容")

        logger.info(f"调用成功！模型返回内容：{content}")

        return content

    def stream_chat(self, message: str) -> Iterable[str]:
        if not message or not message.strip():
            raise ValueError("message cannot be empty")
        
        logger.info("Starting streaming model call: %s", self.model_name)

        stream = self.client.chat.completions.create(
            model=self.model_name,
            messages=self._build_messages(message),
            temperature=0.2,
            stream=True,    # 设置为流式输出
        )
        # 取出chunk中的内容
        for chunk in stream:
            if not chunk.choices:
                continue    
            content = chunk.choices[0].delta.content
            # 过滤空字符串或none
            if content:
                yield content

        logger.info("Streaming model call completed")

    def _build_messages(self, message: str) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": "你是一个企业私有知识助手，请基于已知信息准确、简洁地回答用户问题。",
            },
            {
                "role": "user",
                "content": message,
            },
        ]