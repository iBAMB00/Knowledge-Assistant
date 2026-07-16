from llm_service import LLMService



class ChatService:
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service