from enum import Enum


class ProcessingJobStage(str, Enum):
    """
    文档处理任务当前业务阶段。
    """

    QUEUED = "queued"
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
