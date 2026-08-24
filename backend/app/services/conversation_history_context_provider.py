"""把持久化 ConversationMessage 映射为模型可见 Context。"""

from sqlalchemy.orm import Session

from app.agent.context_engine import (
    AgentContextItem,
    AgentContextRole,
    AgentContextSource,
)
from app.constants.conversation_message_role import ConversationMessageRole
from app.constants.conversation_mode import ConversationMode
from app.models.database.conversation_message import ConversationMessage
from app.services.conversation_service import ConversationService

class ConversationHistoryContextProvider:
    """读取当前用户自己的 Conversation 历史，并转换为 Agent Context。

    当前用户消息在进入 Runtime 前已经持久化，因此 B3 会从结果尾部精确
    排除这一条“当前 turn user message”，避免同时以 HISTORY + CURRENT_MESSAGE
    两种身份重复发送给模型。B4 再负责 token budget / truncation。
    """

    HISTORY_LIMIT = 200

    def __init__(self, *, conversation_service: ConversationService) -> None:
        self.conversation_service = conversation_service

    def load_records(
        self,
        db: Session,
        *,
        user_id: int,
        conversation_id: int | None,
        knowledge_base_id: int,
        current_message: str,
    ) -> list[ConversationMessage]:
        """读取当前 turn 之前的持久化消息，并保护 Conversation Scope。"""

        if conversation_id is None:
            return []

        normalized_current = current_message.strip()
        if not normalized_current:
            raise ValueError("current_message cannot be empty")

        self.conversation_service.ensure_chat_scope(
            db=db,
            user_id=user_id,
            conversation_id=conversation_id,
            mode=ConversationMode.AGENT,
            knowledge_base_id=knowledge_base_id,
        )

        messages = self.conversation_service.list_messages(
            db,
            user_id=user_id,
            conversation_id=conversation_id,
            limit=self.HISTORY_LIMIT,
        )

        # Chat API 在 Runtime 前已经 durable 写入当前 user message。只移除
        # 最后一条且必须 role/content 同时匹配，避免误删更早的同文问题。
        if messages:
            last = messages[-1]
            if (
                last.role == ConversationMessageRole.USER.value
                and last.content.strip() == normalized_current
            ):
                messages = messages[:-1]

        return list(messages)

    @staticmethod
    def to_context_items(
        messages: list[ConversationMessage],
        *,
        after_message_id: int | None = None,
    ) -> tuple[AgentContextItem, ...]:
        """把消息记录映射为 History Context，可从 Summary 边界后开始。"""

        items: list[AgentContextItem] = []
        for message in messages:
            if (
                after_message_id is not None
                and message.id <= after_message_id
            ):
                continue
            if message.role == ConversationMessageRole.USER.value:
                role = AgentContextRole.USER
            elif message.role == ConversationMessageRole.ASSISTANT.value:
                role = AgentContextRole.ASSISTANT
            else:
                raise ValueError(
                    f"unsupported conversation message role: {message.role}"
                )

            content = message.content.strip()
            if not content:
                continue

            items.append(
                AgentContextItem(
                    role=role,
                    source=AgentContextSource.CONVERSATION_HISTORY,
                    content=content,
                    source_id=f"conversation_message:{message.id}",
                )
            )

        return tuple(items)

    def load(
        self,
        db: Session,
        *,
        user_id: int,
        conversation_id: int | None,
        knowledge_base_id: int,
        current_message: str,
        after_message_id: int | None = None,
    ) -> tuple[AgentContextItem, ...]:
        """兼容 B3 入口，并允许 B5 从 Summary 覆盖边界之后加载 History。"""

        messages = self.load_records(
            db,
            user_id=user_id,
            conversation_id=conversation_id,
            knowledge_base_id=knowledge_base_id,
            current_message=current_message,
        )
        return self.to_context_items(
            messages,
            after_message_id=after_message_id,
        )
