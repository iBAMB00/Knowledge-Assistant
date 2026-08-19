from enum import Enum


class AgentRuntime(str, Enum):
    """HTTP 层允许显式选择的 Agent Runtime。"""

    NATIVE = "native"
    LANGCHAIN = "langchain"
