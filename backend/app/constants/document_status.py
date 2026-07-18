from enum import Enum

class DocumentStatus(str, Enum):
    """
    文档生命周期状态。

    用于管理文档从上传到完成处理的状态流转。
    """

    UPLOADED = "uploaded"

    PARSING = "parsing"

    PARSED = "parsed"

    CHUNKING = "chunking"

    EMBEDDING = "embedding"

    COMPLETED = "completed"

    FAILED = "failed"