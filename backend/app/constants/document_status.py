from enum import Enum

class DocumentStatus(str, Enum):
    """
    文档状态枚举。
    """

    UPLOADED = "uploaded"

    PARSING = "parsing"

    PARSED = "parsed"

    CHUNKING = "chunking"

    EMBEDDING = "embedding"

    COMPLETED = "completed"

    FAILED = "failed"