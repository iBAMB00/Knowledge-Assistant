from enum import Enum


class MemoryType(str, Enum):
    """Conversation-derived Memory 的第一版稳定类型。"""

    FACT = "fact"
    PREFERENCE = "preference"
    CONSTRAINT = "constraint"
    DECISION = "decision"
    GOAL = "goal"
