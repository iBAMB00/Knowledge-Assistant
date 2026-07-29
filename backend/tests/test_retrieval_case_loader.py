import json
from pathlib import Path

import pytest

from app.services.evaluation.retrieval_case_loader import (
    RetrievalCaseLoader,
)


def write_json(
    file_path: Path,
    data: object,
) -> None:
    """
    写入测试JSON文件。
    """

    file_path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_load_returns_valid_evaluation_cases(
    tmp_path: Path,
) -> None:
    """
    验证合法问题集可以被加载。
    """

    case_file = tmp_path / "cases.json"

    write_json(
        case_file,
        [
            {
                "case_id": "case-001",
                "question": "如何重置密码？",
                "expected_document_ids": [1],
            },
            {
                "case_id": "case-002",
                "question": "权限如何配置？",
                "expected_document_ids": [
                    1,
                    2,
                ],
                "document_id": 1,
            },
        ],
    )

    cases = RetrievalCaseLoader.load(
        case_file
    )

    assert len(cases) == 2

    assert cases[0].case_id == "case-001"
    assert cases[0].question == (
        "如何重置密码？"
    )
    assert (
        cases[0].expected_document_ids
        == [1]
    )

    assert cases[1].document_id == 1


def test_load_rejects_missing_file(
    tmp_path: Path,
) -> None:
    """
    验证不存在的问题集文件被拒绝。
    """

    missing_file = (
        tmp_path / "missing.json"
    )

    with pytest.raises(
        FileNotFoundError,
        match="does not exist",
    ):
        RetrievalCaseLoader.load(
            missing_file
        )


def test_load_rejects_invalid_json(
    tmp_path: Path,
) -> None:
    """
    验证非法JSON被拒绝。
    """

    case_file = tmp_path / "cases.json"

    case_file.write_text(
        "{invalid json",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="invalid JSON",
    ):
        RetrievalCaseLoader.load(
            case_file
        )


def test_load_rejects_empty_case_list(
    tmp_path: Path,
) -> None:
    """
    验证空评估问题集被拒绝。
    """

    case_file = tmp_path / "cases.json"

    write_json(
        case_file,
        [],
    )

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        RetrievalCaseLoader.load(
            case_file
        )


def test_load_rejects_duplicate_case_id(
    tmp_path: Path,
) -> None:
    """
    验证重复的case_id被拒绝。
    """

    case_file = tmp_path / "cases.json"

    write_json(
        case_file,
        [
            {
                "case_id": "case-001",
                "question": "问题一",
                "expected_document_ids": [1],
            },
            {
                "case_id": " case-001 ",
                "question": "问题二",
                "expected_document_ids": [2],
            },
        ],
    )

    with pytest.raises(
        ValueError,
        match="duplicate case_id",
    ):
        RetrievalCaseLoader.load(
            case_file
        )


def test_load_rejects_blank_question(
    tmp_path: Path,
) -> None:
    """
    验证空白问题被拒绝。
    """

    case_file = tmp_path / "cases.json"

    write_json(
        case_file,
        [
            {
                "case_id": "case-001",
                "question": "   ",
                "expected_document_ids": [1],
            }
        ],
    )

    with pytest.raises(
        ValueError,
        match="question cannot be empty",
    ):
        RetrievalCaseLoader.load(
            case_file
        )


def test_load_rejects_duplicate_expected_documents(
    tmp_path: Path,
) -> None:
    """
    验证重复的预期文档ID被拒绝。
    """

    case_file = tmp_path / "cases.json"

    write_json(
        case_file,
        [
            {
                "case_id": "case-001",
                "question": "测试问题",
                "expected_document_ids": [
                    1,
                    1,
                ],
            }
        ],
    )

    with pytest.raises(
        ValueError,
        match="cannot contain duplicates",
    ):
        RetrievalCaseLoader.load(
            case_file
        )