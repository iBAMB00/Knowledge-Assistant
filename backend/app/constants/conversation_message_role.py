from enum import Enum


class ConversationMessageRole(str, Enum):
    """
    用户可见 Conversation Message 的最小角色集合。

    System Prompt、Tool Call 与 Tool Result 属于运行时数据，
    不作为普通用户可见历史消息写入该 Contract。
    """

    USER = "user"
    ASSISTANT = "assistant"
