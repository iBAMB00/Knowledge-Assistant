import hashlib
import json
from json import JSONDecodeError
from pathlib import Path

from pydantic import ValidationError

from app.schemas.agent_evaluation import (
    AgentEvaluationDataset,
    AgentEvaluationDatasetReference,
    AgentEvaluationObservationSet,
)


class AgentEvaluationCaseLoader:
    """Agent Eval Dataset / Observation 文件加载器。"""

    @classmethod
    def load_dataset(
        cls,
        file_path: str | Path,
    ) -> AgentEvaluationDataset:
        raw_data = cls._load_json(file_path)
        try:
            return AgentEvaluationDataset.model_validate(raw_data)
        except ValidationError as exc:
            raise ValueError(
                f"agent evaluation dataset is invalid: {exc}"
            ) from exc

    @classmethod
    def load_observations(
        cls,
        file_path: str | Path,
    ) -> AgentEvaluationObservationSet:
        raw_data = cls._load_json(file_path)
        try:
            return AgentEvaluationObservationSet.model_validate(raw_data)
        except ValidationError as exc:
            raise ValueError(
                f"agent evaluation observations are invalid: {exc}"
            ) from exc

    @staticmethod
    def build_reference(
        dataset: AgentEvaluationDataset,
        file_path: str | Path,
    ) -> AgentEvaluationDatasetReference:
        resolved_path = Path(file_path).resolve()
        if not resolved_path.is_file():
            raise ValueError(
                "agent evaluation dataset path must be a file: "
                f"{resolved_path}"
            )

        return AgentEvaluationDatasetReference(
            schema_version=dataset.schema_version,
            dataset_id=dataset.dataset_id,
            dataset_version=dataset.dataset_version,
            source_path=str(resolved_path),
            source_sha256=hashlib.sha256(
                resolved_path.read_bytes()
            ).hexdigest(),
            total_cases=len(dataset.cases),
        )

    @staticmethod
    def validate_observation_coverage(
        dataset: AgentEvaluationDataset,
        observations: AgentEvaluationObservationSet,
    ) -> None:
        if observations.dataset_id != dataset.dataset_id:
            raise ValueError("observation dataset_id does not match dataset")
        if observations.dataset_version != dataset.dataset_version:
            raise ValueError(
                "observation dataset_version does not match dataset"
            )

        expected_case_ids = {case.case_id for case in dataset.cases}
        observed_case_ids = {
            observation.case_id
            for observation in observations.observations
        }

        missing = expected_case_ids - observed_case_ids
        extra = observed_case_ids - expected_case_ids
        if missing or extra:
            raise ValueError(
                "observation coverage does not match dataset: "
                f"missing={sorted(missing)} extra={sorted(extra)}"
            )

    @staticmethod
    def _load_json(file_path: str | Path) -> object:
        resolved_path = Path(file_path)
        if not resolved_path.exists():
            raise FileNotFoundError(
                f"evaluation file does not exist: {resolved_path}"
            )
        if not resolved_path.is_file():
            raise ValueError(
                f"evaluation path must be a file: {resolved_path}"
            )

        try:
            return json.loads(resolved_path.read_text(encoding="utf-8"))
        except JSONDecodeError as exc:
            raise ValueError(
                f"evaluation file contains invalid JSON: {exc.msg}"
            ) from exc
