from collections.abc import Iterator

from app.services.llm_service import LLMService

'''
聊天服务
负责聊天业务入口，调用LLM模型服务
'''

class ChatService:
    def __init__(self, llm_service: LLMService) -> None:
        self.llm_service = llm_service

    def chat(self, message: str) -> str:
        return self.llm_service.chat(message)

    def stream_chat(self, message: str) -> Iterator[str]:
        yield from self.llm_service.stream_chat(message)