import re


SOURCE_REF_PATTERN = re.compile(
    r"\[source:(?P<source_ref>[A-Za-z0-9_.:-]+)\]"
)


def build_knowledge_source_ref(*, document_id: int, chunk_id: int) -> str:
    """为知识检索证据生成稳定、无正文的运行时来源引用。"""

    if document_id <= 0:
        raise ValueError("document_id must be greater than 0")
    if chunk_id <= 0:
        raise ValueError("chunk_id must be greater than 0")

    return f"doc:{document_id}:chunk:{chunk_id}"


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
