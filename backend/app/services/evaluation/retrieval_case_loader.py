import json
from json import JSONDecodeError
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from app.schemas.retrieval_evaluation import (
    RetrievalEvaluationCase,
)


class RetrievalCaseLoader:
    """
    检索评估问题集加载器。

    负责：
    - 从JSON文件读取评估问题
    - 使用Pydantic校验数据结构
    - 校验问题内容和用例标识
    - 校验case_id唯一性

    不负责：
    - 执行检索
    - 计算评估指标
    - 修改评估数据
    """

    _cases_adapter = TypeAdapter(
        list[RetrievalEvaluationCase]
    )

    @classmethod
    def load(
        cls,
        file_path: str | Path,
    ) -> list[RetrievalEvaluationCase]:
        """
        从JSON文件加载检索评估问题集。
        """

        resolved_path = Path(file_path)

        if not resolved_path.exists():
            raise FileNotFoundError(
                "evaluation case file does not exist: "
                f"{resolved_path}"
            )

        if not resolved_path.is_file():
            raise ValueError(
                "evaluation case path must be a file: "
                f"{resolved_path}"
            )

        try:
            raw_data = json.loads(
                resolved_path.read_text(
                    encoding="utf-8"
                )
            )

        except JSONDecodeError as exc:
            raise ValueError(
                "evaluation case file contains "
                f"invalid JSON: {exc.msg}"
            ) from exc

        try:
            cases = cls._cases_adapter.validate_python(
                raw_data
            )

        except ValidationError as exc:
            raise ValueError(
                "evaluation case data is invalid: "
                f"{exc}"
            ) from exc

        cls._validate_cases(cases)

        return cases

    @staticmethod
    def _validate_cases(
        cases: list[RetrievalEvaluationCase],
    ) -> None:
        """
        校验评估问题的业务约束。
        """

        if not cases:
            raise ValueError(
                "evaluation case file cannot be empty"
            )

        seen_case_ids: set[str] = set()

        for case in cases:
            normalized_case_id = (
                case.case_id.strip()
            )

            if not normalized_case_id:
                raise ValueError(
                    "case_id cannot be empty"
                )

            if normalized_case_id in seen_case_ids:
                raise ValueError(
                    "duplicate case_id: "
                    f"{normalized_case_id}"
                )

            if not case.question.strip():
                raise ValueError(
                    "question cannot be empty: "
                    f"case_id={normalized_case_id}"
                )

            expected_document_ids = (
                case.expected_document_ids
            )

            if (
                len(set(expected_document_ids))
                != len(expected_document_ids)
            ):
                raise ValueError(
                    "expected_document_ids "
                    "cannot contain duplicates: "
                    f"case_id={normalized_case_id}"
                )

            seen_case_ids.add(
                normalized_case_id
            )