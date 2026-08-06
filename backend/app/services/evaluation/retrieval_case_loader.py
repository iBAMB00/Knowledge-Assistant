import hashlib
import json
from json import JSONDecodeError
from pathlib import Path

from pydantic import ValidationError

from app.schemas.retrieval_evaluation import (
    RetrievalEvaluationDataset,
    RetrievalEvaluationDatasetReference,
)


class RetrievalCaseLoader:
    """
    检索评估数据集加载器。

    负责：
    - 从JSON文件读取版本化评估数据集
    - 使用Pydantic校验结构和业务契约
    - 生成可写入报告的数据集来源快照

    不负责：
    - 校验数据库中的文档和Chunk是否存在
    - 执行检索
    - 计算评估指标
    """

    @classmethod
    def load(
        cls,
        file_path: str | Path,
    ) -> RetrievalEvaluationDataset:
        """从JSON文件加载检索评估数据集。"""

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
            return RetrievalEvaluationDataset.model_validate(
                raw_data
            )

        except ValidationError as exc:
            raise ValueError(
                "evaluation case data is invalid: "
                f"{exc}"
            ) from exc

    @staticmethod
    def build_reference(
        dataset: RetrievalEvaluationDataset,
        file_path: str | Path,
    ) -> RetrievalEvaluationDatasetReference:
        """生成评估报告使用的数据集来源快照。"""

        resolved_path = Path(file_path).resolve()

        if not resolved_path.is_file():
            raise ValueError(
                "evaluation case path must be a file: "
                f"{resolved_path}"
            )

        source_sha256 = hashlib.sha256(
            resolved_path.read_bytes()
        ).hexdigest()

        return RetrievalEvaluationDatasetReference(
            schema_version=dataset.schema_version,
            dataset_id=dataset.dataset_id,
            dataset_version=dataset.dataset_version,
            source_path=str(resolved_path),
            source_sha256=source_sha256,
            strict_corpus=dataset.strict_corpus,
            corpus_document_ids=[
                document.document_id
                for document in dataset.corpus_documents
            ],
            total_cases=len(dataset.cases),
        )
