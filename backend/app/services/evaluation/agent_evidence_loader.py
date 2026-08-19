from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.agent.evidence import parse_knowledge_source_ref
from app.repositories.document_chunk_repository import DocumentChunkRepository
from app.services.evaluation.groundedness_judge import GroundednessEvidence


@dataclass(frozen=True)
class AgentEvidenceLoadResult:
    """Eval-only 证据加载结果；正文只保留在当前进程内。"""

    evidence: tuple[GroundednessEvidence, ...]
    missing_source_refs: tuple[str, ...]


class AgentEvaluationEvidenceLoader:
    """
    根据 D3.1 source_ref 从 SQL Chunk 恢复 Groundedness Judge 证据。

    加载始终限制在当前 KnowledgeBase；不会把正文写入 AgentRun、SSE、
    AgentToolCall 或 Eval Observation。Parent-Child 模式开启时，按 v1.0
    RetrievalService 的行为把 Child 命中恢复为 Parent 正文。
    """

    def __init__(
        self,
        *,
        chunk_repository: DocumentChunkRepository,
        parent_child_enabled: bool,
    ) -> None:
        self.chunk_repository = chunk_repository
        self.parent_child_enabled = parent_child_enabled

    def load(
        self,
        *,
        db: Session,
        knowledge_base_id: int,
        source_refs: list[str],
    ) -> AgentEvidenceLoadResult:
        if knowledge_base_id <= 0:
            raise ValueError("knowledge_base_id must be greater than 0")

        parsed_by_ref = {
            source_ref: parse_knowledge_source_ref(source_ref)
            for source_ref in source_refs
        }
        valid_refs = {
            source_ref: parsed
            for source_ref, parsed in parsed_by_ref.items()
            if parsed is not None
        }
        chunk_ids = sorted({
            parsed.chunk_id
            for parsed in valid_refs.values()
        })

        chunks = self.chunk_repository.find_by_ids(
            db=db,
            chunk_ids=chunk_ids,
            knowledge_base_id=knowledge_base_id,
        )
        chunks_by_id = {chunk.id: chunk for chunk in chunks}
        document_ids = self.chunk_repository.find_document_ids_by_chunk_ids(
            db=db,
            chunk_ids=list(chunks_by_id),
        )

        parent_chunks_by_id = {}
        if self.parent_child_enabled:
            parent_ids = sorted({
                chunk.parent_chunk_id
                for chunk in chunks
                if chunk.parent_chunk_id is not None
            })
            parent_chunks = self.chunk_repository.find_by_ids(
                db=db,
                chunk_ids=parent_ids,
                knowledge_base_id=knowledge_base_id,
            )
            parent_chunks_by_id = {
                chunk.id: chunk
                for chunk in parent_chunks
            }

        evidence: list[GroundednessEvidence] = []
        missing: list[str] = []

        for source_ref in source_refs:
            parsed = parsed_by_ref[source_ref]
            if parsed is None:
                missing.append(source_ref)
                continue

            chunk = chunks_by_id.get(parsed.chunk_id)
            if chunk is None:
                missing.append(source_ref)
                continue

            if document_ids.get(chunk.id) != parsed.document_id:
                missing.append(source_ref)
                continue

            evidence_chunk = chunk
            if self.parent_child_enabled:
                if chunk.parent_chunk_id is None:
                    missing.append(source_ref)
                    continue
                evidence_chunk = parent_chunks_by_id.get(chunk.parent_chunk_id)
                if evidence_chunk is None:
                    missing.append(source_ref)
                    continue

            content = evidence_chunk.content.strip()
            if not content:
                missing.append(source_ref)
                continue

            evidence.append(
                GroundednessEvidence(
                    source_ref=source_ref,
                    content=content,
                )
            )

        return AgentEvidenceLoadResult(
            evidence=tuple(evidence),
            missing_source_refs=tuple(missing),
        )
