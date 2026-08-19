import re
from dataclasses import dataclass


SOURCE_REF_PATTERN = re.compile(
    r"\[source:(?P<source_ref>[A-Za-z0-9_.:-]+)\]"
)
KNOWLEDGE_SOURCE_REF_PATTERN = re.compile(
    r"^doc:(?P<document_id>[1-9]\d*):chunk:(?P<chunk_id>[1-9]\d*)$"
)


@dataclass(frozen=True)
class KnowledgeSourceRef:
    """结构化知识证据引用，不包含任何企业正文。"""

    document_id: int
    chunk_id: int


def build_knowledge_source_ref(*, document_id: int, chunk_id: int) -> str:
    """为知识检索证据生成稳定、无正文的运行时来源引用。"""

    if document_id <= 0:
        raise ValueError("document_id must be greater than 0")
    if chunk_id <= 0:
        raise ValueError("chunk_id must be greater than 0")

    return f"doc:{document_id}:chunk:{chunk_id}"


def parse_knowledge_source_ref(source_ref: str) -> KnowledgeSourceRef | None:
    """解析标准知识 source_ref；非法或其他来源类型返回 None。"""

    normalized = source_ref.strip()
    match = KNOWLEDGE_SOURCE_REF_PATTERN.fullmatch(normalized)
    if match is None:
        return None

    return KnowledgeSourceRef(
        document_id=int(match.group("document_id")),
        chunk_id=int(match.group("chunk_id")),
    )


def extract_source_refs(answer: str) -> list[str]:
    """从最终回答中按出现顺序提取并去重标准 source_ref。"""

    if not answer:
        return []

    refs: list[str] = []
    seen: set[str] = set()

    for match in SOURCE_REF_PATTERN.finditer(answer):
        source_ref = match.group("source_ref")
        if source_ref in seen:
            continue
        seen.add(source_ref)
        refs.append(source_ref)

    return refs
