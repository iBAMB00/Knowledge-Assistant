from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.core.database import SessionLocal
from app.repositories.document_chunk_repository import DocumentChunkRepository
from app.repositories.document_content_repository import DocumentContentRepository
from app.repositories.document_repository import DocumentRepository


DEFAULT_BLUEPRINT_PATH = Path(
    "evaluation/retrieval_cases_blueprint_v2.json"
)
DEFAULT_OUTPUT_PATH = Path(
    "evaluation/retrieval_cases_v2.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "将文件名和证据文本蓝图转换为可直接运行的检索评估数据集。"
            "数据库连接使用当前应用 DATABASE_URL，支持 SQLite/PostgreSQL。"
        )
    )
    parser.add_argument(
        "--blueprint",
        type=Path,
        default=DEFAULT_BLUEPRINT_PATH,
        help="评估用例蓝图路径。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="生成的正式评估数据集路径。",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def choose_canonical_chunk(
    rows: list[Any],
    evidence_text: str,
) -> Any:
    """
    重叠 Parent 可能同时包含同一证据句。
    选择证据位于切片中更居中的 Parent，降低边界重叠带来的随机性。
    """

    def score(row: Any) -> tuple[int, int, int]:
        content = row.content
        position = content.find(evidence_text)
        margin = min(
            position,
            len(content) - (position + len(evidence_text)),
        )
        return (
            margin,
            -len(content),
            -row.id,
        )

    return max(rows, key=score)


def build_dataset(
    db: Session,
    blueprint: dict[str, Any],
) -> tuple[dict[str, Any], int]:
    """使用当前业务数据库重新解析 document_id / Parent chunk_id。"""

    document_repository = DocumentRepository()
    content_repository = DocumentContentRepository()
    chunk_repository = DocumentChunkRepository()

    documents = document_repository.find_all(db)
    documents_by_filename = {
        document.filename: document
        for document in documents
    }

    target_filenames = [
        item["filename"]
        for item in blueprint["corpus_files"]
    ] + list(blueprint.get("retain_filenames", []))

    missing_documents = [
        filename
        for filename in target_filenames
        if filename not in documents_by_filename
    ]
    if missing_documents:
        raise RuntimeError(
            "数据库缺少语料文档: "
            + ", ".join(missing_documents)
        )

    incomplete_documents = [
        filename
        for filename in target_filenames
        if documents_by_filename[filename].status != "completed"
    ]
    if incomplete_documents:
        raise RuntimeError(
            "以下文档尚未完成处理，请先完成 full_pipeline: "
            + ", ".join(incomplete_documents)
        )

    if blueprint.get("strict_corpus", True):
        unexpected_documents = sorted(
            set(documents_by_filename)
            - set(target_filenames)
        )
        if unexpected_documents:
            raise RuntimeError(
                "严格语料模式下数据库存在额外文档: "
                + ", ".join(unexpected_documents)
            )

    target_document_ids = [
        documents_by_filename[filename].id
        for filename in target_filenames
    ]
    contents_by_document_id = content_repository.find_by_document_ids(
        db=db,
        document_ids=target_document_ids,
    )

    missing_contents = [
        filename
        for filename in target_filenames
        if documents_by_filename[filename].id
        not in contents_by_document_id
    ]
    if missing_contents:
        raise RuntimeError(
            "数据库缺少解析正文: "
            + ", ".join(missing_contents)
        )

    parent_chunks_by_document_id: dict[int, list[Any]] = {}
    for filename in target_filenames:
        document_id = documents_by_filename[filename].id
        content = contents_by_document_id[document_id]
        chunks = chunk_repository.find_by_document_content_id(
            db=db,
            document_content_id=content.id,
        )
        parents = [
            chunk
            for chunk in chunks
            if chunk.parent_chunk_id is None
        ]
        if not parents:
            raise RuntimeError(
                f"文档没有 Parent Chunk: {filename}"
            )
        parent_chunks_by_document_id[document_id] = parents

    corpus_documents: list[dict[str, Any]] = []
    for filename in target_filenames:
        document = documents_by_filename[filename]
        content = contents_by_document_id[document.id]
        corpus_documents.append({
            "document_id": document.id,
            "filename": filename,
            "content_sha256": hashlib.sha256(
                content.content.encode("utf-8")
            ).hexdigest(),
        })

    materialized_cases: list[dict[str, Any]] = []
    ambiguous_anchor_count = 0

    for case in blueprint["cases"]:
        expected_filenames = case.get(
            "expected_filenames",
            [],
        )
        expected_document_ids = [
            documents_by_filename[filename].id
            for filename in expected_filenames
        ]

        expected_chunk_ids: list[int] = []

        for evidence_text in case.get("evidence_texts", []):
            rows = [
                chunk
                for document_id in expected_document_ids
                for chunk in parent_chunks_by_document_id[document_id]
                if evidence_text in chunk.content
            ]

            if not rows:
                raise RuntimeError(
                    f"用例 {case['case_id']} 的证据文本未命中任何 Parent Chunk: "
                    f"{evidence_text}"
                )

            if len(rows) > 1:
                ambiguous_anchor_count += 1

            chosen_chunk = choose_canonical_chunk(
                rows=rows,
                evidence_text=evidence_text,
            )
            expected_chunk_ids.append(chosen_chunk.id)

        expected_chunk_ids = list(
            dict.fromkeys(expected_chunk_ids)
        )

        materialized_case = {
            "case_id": case["case_id"],
            "question": case["question"],
            "category": case["category"],
            "difficulty": case["difficulty"],
            "should_retrieve": case["should_retrieve"],
            "expected_document_ids": expected_document_ids,
            "expected_chunk_ids": expected_chunk_ids,
            "relevant_texts": case.get("evidence_texts", []),
            "keywords": case.get("keywords", []),
            "tags": case.get("tags", []),
        }

        filter_filename = case.get("document_filter_filename")
        if filter_filename:
            materialized_case["document_id"] = (
                documents_by_filename[filter_filename].id
            )

        if case.get("notes"):
            materialized_case["notes"] = case["notes"]

        materialized_cases.append(materialized_case)

    dataset = {
        "schema_version": "1.0",
        "dataset_id": blueprint["dataset_id"],
        "dataset_version": blueprint["dataset_version"],
        "description": blueprint["description"],
        "strict_corpus": blueprint.get("strict_corpus", True),
        "corpus_documents": corpus_documents,
        "cases": materialized_cases,
    }

    return dataset, ambiguous_anchor_count


def main() -> None:
    args = parse_args()
    blueprint_path = args.blueprint.resolve()
    output_path = args.output.resolve()
    blueprint = load_json(blueprint_path)

    with SessionLocal() as db:
        dataset, ambiguous_anchor_count = build_dataset(
            db=db,
            blueprint=blueprint,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            dataset,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "Evaluation dataset generated: "
        f"documents={len(dataset['corpus_documents'])}, "
        f"cases={len(dataset['cases'])}, "
        f"ambiguous_anchors={ambiguous_anchor_count}"
    )
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
