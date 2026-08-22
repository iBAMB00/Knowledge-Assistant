from functools import lru_cache

from app.repositories.conversation_message_repository import (
    ConversationMessageRepository,
)
from app.repositories.conversation_repository import ConversationRepository
from app.repositories.document_repository import DocumentRepository
from app.repositories.knowledge_base_repository import KnowledgeBaseRepository
from app.services.conversation_service import ConversationService
from app.services.knowledge_base_access_policy import KnowledgeBaseAccessPolicy


@lru_cache
def get_conversation_service() -> ConversationService:
    """构建 Conversation / Message 共用的持久化服务。"""

    return ConversationService(
        conversation_repository=ConversationRepository(),
        message_repository=ConversationMessageRepository(),
        access_policy=KnowledgeBaseAccessPolicy(
            knowledge_base_repository=KnowledgeBaseRepository(),
            document_repository=DocumentRepository(),
        ),
    )
