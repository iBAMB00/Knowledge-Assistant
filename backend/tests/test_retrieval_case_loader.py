import hashlib
import json
from pathlib import Path

import pytest

from app.schemas.retrieval_evaluation import (
    RetrievalCaseCategory,
    RetrievalCaseDifficulty,
)
from app.services.evaluation.retrieval_case_loader import (
    RetrievalCaseLoader,
)


def write_json(
    file_path: Path,
    data: object,
) -> None:
    """写入测试JSON文件。"""

    file_path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def build_dataset_data() -> dict:
    """创建最小合法版本化评估数据集。"""

    return {
        "schema_version": "1.0",
        "dataset_id": "test-retrieval-dataset",
        "dataset_version": "1.0.0",
        "description": "测试检索数据集",
        "strict_corpus": True,
        "corpus_documents": [
            {
                "document_id": 1,
                "filename": "document.txt",
                "content_sha256": "0" * 64,
            }
        ],
        "cases": [
            {
                "case_id": "case-001",
                "question": "如何重置密码？",
                "category": "procedure",
                "difficulty": "easy",
                "should_retrieve": True,
                "expected_document_ids": [1],
                "expected_chunk_ids": [10],
                "keywords": ["重置密码"],
            }
        ],
    }


def test_load_returns_valid_versioned_dataset(
    tmp_path: Path,
) -> None:
    """验证版本化评估数据集可以被加载。"""

    case_file = tmp_path / "cases.json"
    write_json(case_file, build_dataset_data())

    dataset = RetrievalCaseLoader.load(
        case_file
    )

    assert dataset.schema_version == "1.0"
    assert dataset.dataset_id == (
        "test-retrieval-dataset"
    )
    assert dataset.dataset_version == "1.0.0"
    assert len(dataset.cases) == 1

    case = dataset.cases[0]

    assert case.case_id == "case-001"
    assert (
        case.category
        == RetrievalCaseCategory.PROCEDURE
    )
    assert (
        case.difficulty
        == RetrievalCaseDifficulty.EASY
    )
    assert case.expected_chunk_ids == [10]


def test_build_reference_records_source_hash(
    tmp_path: Path,
) -> None:
    """验证报告可以记录数据集文件哈希。"""

    case_file = tmp_path / "cases.json"
    write_json(case_file, build_dataset_data())

    dataset = RetrievalCaseLoader.load(
        case_file
    )
    reference = (
        RetrievalCaseLoader.build_reference(
            dataset=dataset,
            file_path=case_file,
        )
    )

    assert reference.total_cases == 1
    assert reference.corpus_document_ids == [1]
    assert reference.source_sha256 == (
        hashlib.sha256(
            case_file.read_bytes()
        ).hexdigest()
    )


def test_load_rejects_missing_file(
    tmp_path: Path,
) -> None:
    """验证不存在的数据集文件被拒绝。"""

    with pytest.raises(
        FileNotFoundError,
        match="does not exist",
    ):
        RetrievalCaseLoader.load(
            tmp_path / "missing.json"
        )


def test_load_rejects_invalid_json(
    tmp_path: Path,
) -> None:
    """验证非法JSON被拒绝。"""

    case_file = tmp_path / "cases.json"
    case_file.write_text(
        "{invalid json",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="invalid JSON",
    ):
        RetrievalCaseLoader.load(case_file)


def test_load_rejects_legacy_case_list(
    tmp_path: Path,
) -> None:
    """验证缺少版本和语料清单的旧列表格式被拒绝。"""

    case_file = tmp_path / "cases.json"
    write_json(
        case_file,
        [
            {
                "case_id": "case-001",
                "question": "测试问题",
                "expected_document_ids": [1],
            }
        ],
    )

    with pytest.raises(
        ValueError,
        match="evaluation case data is invalid",
    ):
        RetrievalCaseLoader.load(case_file)


def test_load_rejects_duplicate_case_id(
    tmp_path: Path,
) -> None:
    """验证规范化后重复的case_id被拒绝。"""

    data = build_dataset_data()
    duplicated_case = {
        **data["cases"][0],
        "case_id": " case-001 ",
    }
    data["cases"].append(duplicated_case)

    case_file = tmp_path / "cases.json"
    write_json(case_file, data)

    with pytest.raises(
        ValueError,
        match="case_id cannot contain duplicates",
    ):
        RetrievalCaseLoader.load(case_file)


def test_load_rejects_duplicate_expected_ids(
    tmp_path: Path,
) -> None:
    """验证重复预期ID被拒绝。"""

    data = build_dataset_data()
    data["cases"][0][
        "expected_document_ids"
    ] = [1, 1]

    case_file = tmp_path / "cases.json"
    write_json(case_file, data)

    with pytest.raises(
        ValueError,
        match="expected IDs cannot contain duplicates",
    ):
        RetrievalCaseLoader.load(case_file)


def test_load_supports_no_answer_case(
    tmp_path: Path,
) -> None:
    """验证无答案用例使用显式契约。"""

    data = build_dataset_data()
    data["cases"] = [
        {
            "case_id": "no-answer-001",
            "question": "产品价格是多少？",
            "category": "no_answer",
            "difficulty": "easy",
            "should_retrieve": False,
            "expected_document_ids": [],
            "expected_chunk_ids": [],
        }
    ]

    case_file = tmp_path / "cases.json"
    write_json(case_file, data)

    dataset = RetrievalCaseLoader.load(
        case_file
    )

    assert dataset.cases[0].should_retrieve is False


def test_load_rejects_no_answer_with_expected_documents(
    tmp_path: Path,
) -> None:
    """验证无答案用例不能同时标注目标文档。"""

    data = build_dataset_data()
    data["cases"][0].update({
        "category": "no_answer",
        "should_retrieve": False,
    })

    case_file = tmp_path / "cases.json"
    write_json(case_file, data)

    with pytest.raises(
        ValueError,
        match=(
            "no-answer case cannot define "
            "expected_document_ids"
        ),
    ):
        RetrievalCaseLoader.load(case_file)


def test_load_rejects_document_outside_manifest(
    tmp_path: Path,
) -> None:
    """验证用例不能引用语料清单之外的文档。"""

    data = build_dataset_data()
    data["cases"][0][
        "expected_document_ids"
    ] = [2]

    case_file = tmp_path / "cases.json"
    write_json(case_file, data)

    with pytest.raises(
        ValueError,
        match="outside the corpus manifest",
    ):
        RetrievalCaseLoader.load(case_file)
