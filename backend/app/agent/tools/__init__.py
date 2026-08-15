"""Agent Tool adapters."""

from app.agent.tools.document_get import (
    DocumentGetInput,
    DocumentGetOutput,
    DocumentGetTool,
)
from app.agent.tools.document_list import (
    DocumentListInput,
    DocumentListOutput,
    DocumentListTool,
)
from app.agent.tools.knowledge_base_list import (
    KnowledgeBaseListInput,
    KnowledgeBaseListOutput,
    KnowledgeBaseListTool,
)
from app.agent.tools.knowledge_search import (
    KnowledgeSearchInput,
    KnowledgeSearchOutput,
    KnowledgeSearchTool,
)
from app.agent.tools.processing_job_get import (
    ProcessingJobGetInput,
    ProcessingJobGetOutput,
    ProcessingJobGetTool,
)

__all__ = [
    "DocumentGetInput",
    "DocumentGetOutput",
    "DocumentGetTool",
    "DocumentListInput",
    "DocumentListOutput",
    "DocumentListTool",
    "KnowledgeBaseListInput",
    "KnowledgeBaseListOutput",
    "KnowledgeBaseListTool",
    "KnowledgeSearchInput",
    "KnowledgeSearchOutput",
    "KnowledgeSearchTool",
    "ProcessingJobGetInput",
    "ProcessingJobGetOutput",
    "ProcessingJobGetTool",
]
