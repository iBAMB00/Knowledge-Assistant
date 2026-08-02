from enum import Enum


class ProcessingJobType(str, Enum):
    """
    文档处理任务类型。
    """

    DOCUMENT_PROCESSING = "document_processing"
    EMBEDDING = "embedding"
    FULL_PIPELINE = "full_pipeline"