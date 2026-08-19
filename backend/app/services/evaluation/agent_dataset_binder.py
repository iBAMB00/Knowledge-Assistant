from __future__ import annotations

import json
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.schemas.agent_evaluation import (
    AgentEvaluationDataset,
    AgentEvaluationFixtureManifest,
)


class AgentEvaluationDatasetBinder:
    """把版本化 Agent Eval 模板中的 Fixture 占位值绑定为当前环境真实 ID。"""

    @classmethod
    def bind(
        cls,
        *,
        dataset: AgentEvaluationDataset,
        manifest: AgentEvaluationFixtureManifest,
    ) -> AgentEvaluationDataset:
        placeholders = dataset.fixture_placeholders
        if not placeholders:
            return dataset

        missing_bindings = set(placeholders) - set(manifest.bindings)
        if missing_bindings:
            raise ValueError(
                "agent evaluation fixture bindings are incomplete: "
                f"missing={sorted(missing_bindings)}"
            )

        replacements = {
            placeholder_value: manifest.bindings[placeholder_name]
            for placeholder_name, placeholder_value in placeholders.items()
        }

        raw = dataset.model_dump(mode="python")
        raw = cls._replace_placeholders(raw, replacements)
        raw["fixture_placeholders"] = dict(placeholders)
        raw["fixture_bindings"] = {
            key: manifest.bindings[key]
            for key in placeholders
        }
        return AgentEvaluationDataset.model_validate(raw)

    @staticmethod
    def ensure_live_ready(dataset: AgentEvaluationDataset) -> None:
        """有 Fixture 占位符的数据集必须完成环境绑定后才能跑 Live Eval。"""

        if not dataset.fixture_placeholders:
            return

        if set(dataset.fixture_bindings) != set(dataset.fixture_placeholders):
            raise ValueError(
                "agent evaluation dataset requires fixture binding before live run"
            )

    @staticmethod
    def load_manifest(
        file_path: str | Path,
    ) -> AgentEvaluationFixtureManifest:
        resolved_path = Path(file_path)
        if not resolved_path.exists():
            raise FileNotFoundError(
                f"agent evaluation fixture manifest does not exist: {resolved_path}"
            )
        if not resolved_path.is_file():
            raise ValueError(
                f"agent evaluation fixture manifest must be a file: {resolved_path}"
            )

        try:
            raw = json.loads(resolved_path.read_text(encoding="utf-8"))
            return AgentEvaluationFixtureManifest.model_validate(raw)
        except JSONDecodeError as exc:
            raise ValueError(
                "agent evaluation fixture manifest contains invalid JSON: "
                f"{exc.msg}"
            ) from exc
        except ValidationError as exc:
            raise ValueError(
                f"agent evaluation fixture manifest is invalid: {exc}"
            ) from exc

    @classmethod
    def _replace_placeholders(
        cls,
        value: Any,
        replacements: dict[int, int],
    ) -> Any:
        if isinstance(value, bool):
            return value

        if isinstance(value, int):
            return replacements.get(value, value)

        if isinstance(value, str):
            result = value
            for placeholder, bound_value in replacements.items():
                result = result.replace(str(placeholder), str(bound_value))
            return result

        if isinstance(value, list):
            return [
                cls._replace_placeholders(item, replacements)
                for item in value
            ]

        if isinstance(value, dict):
            return {
                key: cls._replace_placeholders(item, replacements)
                for key, item in value.items()
            }

        return value
