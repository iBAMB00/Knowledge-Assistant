from datetime import datetime
from enum import Enum
from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    field_validator,
    model_validator,
)


RetrievalEvaluationMode = Literal[
    "baseline",
    "optimized",
]


class RetrievalCaseCategory(str, Enum):
    """检索评估用例类别。"""

    EXACT_TERM = "exact_term"
    PARAPHRASE = "paraphrase"
    PROCEDURE = "procedure"
    MULTI_DOCUMENT = "multi_document"
    DISTRACTOR = "distractor"
    DOCUMENT_FILTER = "document_filter"
    NO_ANSWER = "no_answer"


class RetrievalCaseDifficulty(str, Enum):
    """检索评估用例难度。"""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class RetrievalEvaluationDocumentReference(BaseModel):
    """评估语料库中的固定文档引用。"""

    model_config = ConfigDict(extra="forbid")

    document_id: PositiveInt
    filename: str = Field(min_length=1, max_length=255)
    content_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    @field_validator("filename")
    @classmethod
    def normalize_filename(cls, value: str) -> str:
        """移除文件名前后空白。"""

        normalized_value = value.strip()

        if not normalized_value:
            raise ValueError("filename cannot be empty")

        return normalized_value


class RetrievalEvaluationCase(BaseModel):
    """单条检索评估问题。"""

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(
        min_length=1,
        max_length=100,
    )
    question: str = Field(min_length=1)
    category: RetrievalCaseCategory
    difficulty: RetrievalCaseDifficulty
    should_retrieve: bool = True

    expected_document_ids: list[PositiveInt] = Field(
        default_factory=list,
    )
    expected_chunk_ids: list[PositiveInt] = Field(
        default_factory=list,
    )

    relevant_texts: list[str] = Field(
        default_factory=list,
    )
    keywords: list[str] = Field(
        default_factory=list,
    )
    tags: list[str] = Field(
        default_factory=list,
    )

    document_id: PositiveInt | None = None
    notes: str | None = None

    @field_validator("case_id", "question")
    @classmethod
    def normalize_required_text(
        cls,
        value: str,
    ) -> str:
        """移除必填文本前后空白。"""

        normalized_value = value.strip()

        if not normalized_value:
            raise ValueError("value cannot be empty")

        return normalized_value

    @field_validator("notes")
    @classmethod
    def normalize_optional_text(
        cls,
        value: str | None,
    ) -> str | None:
        """规范化可选说明文本。"""

        if value is None:
            return None

        normalized_value = value.strip()

        return normalized_value or None

    @field_validator(
        "relevant_texts",
        "keywords",
        "tags",
    )
    @classmethod
    def normalize_text_list(
        cls,
        values: list[str],
    ) -> list[str]:
        """移除列表文本空白并拒绝空值与重复值。"""

        normalized_values = [
            value.strip()
            for value in values
        ]

        if any(
            not value
            for value in normalized_values
        ):
            raise ValueError(
                "text list cannot contain empty values"
            )

        if (
            len(set(normalized_values))
            != len(normalized_values)
        ):
            raise ValueError(
                "text list cannot contain duplicates"
            )

        return normalized_values

    @field_validator(
        "expected_document_ids",
        "expected_chunk_ids",
    )
    @classmethod
    def validate_unique_ids(
        cls,
        values: list[int],
    ) -> list[int]:
        """拒绝重复的文档或Chunk标注。"""

        if len(set(values)) != len(values):
            raise ValueError(
                "expected IDs cannot contain duplicates"
            )

        return values

    @model_validator(mode="after")
    def validate_case_contract(self) -> Self:
        """校验有答案、无答案和文档过滤契约。"""

        if self.should_retrieve:
            if not self.expected_document_ids:
                raise ValueError(
                    "answerable case must define "
                    "expected_document_ids"
                )

            if (
                self.category
                == RetrievalCaseCategory.NO_ANSWER
            ):
                raise ValueError(
                    "no_answer category must set "
                    "should_retrieve to false"
                )

        else:
            if self.expected_document_ids:
                raise ValueError(
                    "no-answer case cannot define "
                    "expected_document_ids"
                )

            if self.expected_chunk_ids:
                raise ValueError(
                    "no-answer case cannot define "
                    "expected_chunk_ids"
                )

            if (
                self.category
                != RetrievalCaseCategory.NO_ANSWER
            ):
                raise ValueError(
                    "should_retrieve=false requires "
                    "no_answer category"
                )

        if (
            self.category
            == RetrievalCaseCategory.MULTI_DOCUMENT
            and len(self.expected_document_ids) < 2
        ):
            raise ValueError(
                "multi_document case requires at least "
                "two expected documents"
            )

        if (
            self.category
            == RetrievalCaseCategory.DOCUMENT_FILTER
            and self.document_id is None
        ):
            raise ValueError(
                "document_filter case requires document_id"
            )

        if self.document_id is not None:
            invalid_document_ids = [
                document_id
                for document_id in self.expected_document_ids
                if document_id != self.document_id
            ]

            if invalid_document_ids:
                raise ValueError(
                    "document-filtered case can only expect "
                    "the filtered document"
                )

        return self


class RetrievalEvaluationDataset(BaseModel):
    """可版本化、可校验的检索评估数据集。"""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"]
    dataset_id: str = Field(
        min_length=1,
        max_length=100,
    )
    dataset_version: str = Field(
        min_length=1,
        max_length=50,
    )
    description: str = Field(min_length=1)
    strict_corpus: bool = True

    corpus_documents: list[
        RetrievalEvaluationDocumentReference
    ] = Field(min_length=1)

    cases: list[RetrievalEvaluationCase] = Field(
        min_length=1,
    )

    @field_validator(
        "dataset_id",
        "dataset_version",
        "description",
    )
    @classmethod
    def normalize_dataset_text(
        cls,
        value: str,
    ) -> str:
        """规范化数据集文本字段。"""

        normalized_value = value.strip()

        if not normalized_value:
            raise ValueError("value cannot be empty")

        return normalized_value

    @model_validator(mode="after")
    def validate_dataset_contract(self) -> Self:
        """校验语料清单、用例标识和引用关系。"""

        document_ids = [
            document.document_id
            for document in self.corpus_documents
        ]
        filenames = [
            document.filename
            for document in self.corpus_documents
        ]
        case_ids = [
            case.case_id
            for case in self.cases
        ]

        if len(set(document_ids)) != len(document_ids):
            raise ValueError(
                "corpus document IDs cannot contain duplicates"
            )

        if len(set(filenames)) != len(filenames):
            raise ValueError(
                "corpus filenames cannot contain duplicates"
            )

        if len(set(case_ids)) != len(case_ids):
            raise ValueError(
                "case_id cannot contain duplicates"
            )

        corpus_document_ids = set(document_ids)

        for case in self.cases:
            referenced_document_ids = set(
                case.expected_document_ids
            )

            if case.document_id is not None:
                referenced_document_ids.add(
                    case.document_id
                )

            unknown_document_ids = (
                referenced_document_ids
                - corpus_document_ids
            )

            if unknown_document_ids:
                raise ValueError(
                    "case references documents outside "
                    "the corpus manifest: "
                    f"case_id={case.case_id}, "
                    f"document_ids={sorted(unknown_document_ids)}"
                )

        return self


class RetrievalEvaluationDatasetReference(BaseModel):
    """写入评估报告的数据集快照。"""

    schema_version: str
    dataset_id: str
    dataset_version: str
    source_path: str
    source_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )
    strict_corpus: bool
    corpus_document_ids: list[int]
    total_cases: int


class RetrievalEvaluationConfiguration(BaseModel):
    """不包含密钥的评估运行配置快照。"""

    executed_at: datetime
    code_version: str | None

    vector_store_backend: Literal[
        "database",
        "qdrant",
    ]

    embedding_provider: str
    embedding_model: str
    embedding_dimension: PositiveInt

    chunk_strategy: str
    chunk_size: PositiveInt
    chunk_overlap: int = Field(ge=0)

    top_k: PositiveInt
    candidate_k: PositiveInt
    score_threshold: float = Field(
        ge=-1.0,
        le=1.0,
    )
    per_document_limit: PositiveInt


class RetrievalEvaluationCaseResult(BaseModel):
    """单条问题的检索评估结果。"""

    case_id: str
    question: str
    category: RetrievalCaseCategory
    difficulty: RetrievalCaseDifficulty
    should_retrieve: bool
    retrieval_mode: RetrievalEvaluationMode

    expected_document_ids: list[int]
    expected_chunk_ids: list[int]

    retrieved_document_ids: list[int]
    retrieved_chunk_ids: list[int]

    hit: bool
    reciprocal_rank: float
    document_coverage: float
    duplicate_rate: float
    no_answer_false_positive: bool
    latency_ms: float


class RetrievalEvaluationSummary(BaseModel):
    """单种检索模式的汇总指标。"""

    retrieval_mode: RetrievalEvaluationMode

    total_cases: int
    answerable_cases: int
    no_answer_cases: int

    hit_rate_at_k: float
    mean_reciprocal_rank: float
    mean_document_coverage: float
    full_document_coverage_rate_at_k: float
    mean_duplicate_rate: float

    no_answer_accuracy: float
    no_answer_false_positive_rate: float

    average_latency_ms: float


class RetrievalEvaluationRun(BaseModel):
    """单种检索模式的完整评估结果。"""

    summary: RetrievalEvaluationSummary
    cases: list[RetrievalEvaluationCaseResult]


class RetrievalComparisonReport(BaseModel):
    """Baseline与Optimized对比报告。"""

    dataset: RetrievalEvaluationDatasetReference
    configuration: RetrievalEvaluationConfiguration
    baseline: RetrievalEvaluationRun
    optimized: RetrievalEvaluationRun
