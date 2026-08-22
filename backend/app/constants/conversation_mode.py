from enum import Enum


class ConversationMode(str, Enum):
    """用户可见 Conversation 的稳定工作模式。"""

    RAG = "rag"
    AGENT = "agent"
