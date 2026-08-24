from enum import Enum


class MemoryStatus(str, Enum):
    """MemoryItem 生命周期状态。"""

    ACTIVE = "active"
    SUPERSEDED = "superseded"
    FORGOTTEN = "forgotten"
