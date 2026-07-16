from llm_service import LLMService
from typing import Iterable



class ChatService:
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service