from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将文件名和证据文本蓝图转换为可直接运行的检索评估数据集。"
    )
    parser.add_argument(
        "--database",
        default="knowledge_assistant.db",
        help="SQLite 数据库路径。",
    )
    parser.add_argument(
        "--blueprint",
        default="evaluation/retrieval_cases_blueprint_v2.json",
        help="评估用例蓝图路径。",
    )
    parser.add_argument(
        "--output",
        default="evaluation/retrieval_cases_v2.json",
        help="生成的正式评估数据集路径。",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def choose_canonical_chunk(
    rows: list[sqlite3.Row],
    evidence_text: str,
) -> sqlite3.Row:
    """
    重叠切片可能同时包含同一证据句。
    选择证据位于切片中更居中的 Chunk，降低边界重叠带来的随机性。
    """
    def score(row: sqlite3.Row) -> tuple[int, int, int]:
        content = row["content"]
        position = content.find(evidence_text)
        margin = min(
            position,
            len(content) - (position + len(evidence_text)),
        )
        return (
            margin,
            -len(content),
            -row["id"],
        )

    return max(rows, key=score)


def main() -> None:
    args = parse_args()
    database_path = Path(args.database).resolve()
    blueprint_path = Path(args.blueprint).resolve()
    output_path = Path(args.output).resolve()

    blueprint = load_json(blueprint_path)

    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row

    try:
        document_rows = connection.execute(
            """
            SELECT
                d.id,
                d.filename,
                d.status,
                dc.content
            FROM documents AS d
            JOIN document_contents AS dc
              ON dc.document_id = d.id
            ORDER BY d.id
            """
        ).fetchall()

        documents_by_filename = {
            row["filename"]: row
            for row in document_rows
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
            if documents_by_filename[filename]["status"] != "completed"
        ]

        if incomplete_documents:
            raise RuntimeError(
                "以下文档尚未完成向量化，请先执行 embedding/full_pipeline: "
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

        corpus_documents: list[dict[str, Any]] = []

        for filename in target_filenames:
            document = documents_by_filename[filename]
            corpus_documents.append({
                "document_id": document["id"],
                "filename": filename,
                "content_sha256": hashlib.sha256(
                    document["content"].encode("utf-8")
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
                documents_by_filename[filename]["id"]
                for filename in expected_filenames
            ]

            expected_chunk_ids: list[int] = []

            for evidence_text in case.get(
                "evidence_texts",
                [],
            ):
                placeholders = ",".join(
                    "?"
                    for _ in expected_document_ids
                )

                rows = connection.execute(
                    f"""
                    SELECT
                        ch.id,
                        ch.chunk_index,
                        ch.content,
                        dc.document_id
                    FROM document_chunks AS ch
                    JOIN document_contents AS dc
                      ON dc.id = ch.document_content_id
                    WHERE dc.document_id IN ({placeholders})
                      AND instr(ch.content, ?) > 0
                    ORDER BY ch.id
                    """,
                    (
                        *expected_document_ids,
                        evidence_text,
                    ),
                ).fetchall()

                if not rows:
                    raise RuntimeError(
                        f"用例 {case['case_id']} 的证据文本未命中任何 Chunk: "
                        f"{evidence_text}"
                    )

                if len(rows) > 1:
                    ambiguous_anchor_count += 1

                chosen_chunk = choose_canonical_chunk(
                    rows=rows,
                    evidence_text=evidence_text,
                )
                expected_chunk_ids.append(
                    chosen_chunk["id"]
                )

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
                "relevant_texts": case.get(
                    "evidence_texts",
                    [],
                ),
                "keywords": case.get(
                    "keywords",
                    [],
                ),
                "tags": case.get(
                    "tags",
                    [],
                ),
            }

            filter_filename = case.get(
                "document_filter_filename"
            )
            if filter_filename:
                materialized_case["document_id"] = (
                    documents_by_filename[
                        filter_filename
                    ]["id"]
                )

            if case.get("notes"):
                materialized_case["notes"] = (
                    case["notes"]
                )

            materialized_cases.append(
                materialized_case
            )

        dataset = {
            "schema_version": "1.0",
            "dataset_id": blueprint["dataset_id"],
            "dataset_version": blueprint[
                "dataset_version"
            ],
            "description": blueprint["description"],
            "strict_corpus": blueprint.get(
                "strict_corpus",
                True,
            ),
            "corpus_documents": corpus_documents,
            "cases": materialized_cases,
        }

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
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
            f"documents={len(corpus_documents)}, "
            f"cases={len(materialized_cases)}, "
            f"ambiguous_anchors={ambiguous_anchor_count}"
        )
        print(f"Output: {output_path}")

    finally:
        connection.close()


if __name__ == "__main__":
    main()
