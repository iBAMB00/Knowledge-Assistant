from __future__ import annotations

import ast
from pathlib import Path

from app.schemas.stateful_fault_verification import StatefulFaultSuiteDefinition


class StatefulFaultSuiteValidationError(ValueError):
    """Stateful fault suite 文件或 nodeid 定义不合法。"""


class StatefulFaultCaseLoader:
    """读取并验证版本化 Stateful Fault Verification Case。"""

    def load(self, path: Path) -> StatefulFaultSuiteDefinition:
        try:
            content = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise StatefulFaultSuiteValidationError(
                f"cannot read stateful fault suite: {path}"
            ) from exc

        try:
            return StatefulFaultSuiteDefinition.model_validate_json(content)
        except Exception as exc:
            raise StatefulFaultSuiteValidationError(
                f"invalid stateful fault suite: {path}"
            ) from exc

    def validate_nodeids_exist(
        self,
        suite: StatefulFaultSuiteDefinition,
        *,
        project_root: Path,
    ) -> None:
        """静态检查 suite 中的 pytest nodeid，避免验证清单悄悄失效。"""

        function_cache: dict[Path, set[str]] = {}
        missing: list[str] = []

        for case in suite.cases:
            for nodeid in case.required_nodeids:
                file_part, separator, function_part = nodeid.partition("::")
                if not separator or not function_part:
                    missing.append(nodeid)
                    continue

                file_path = project_root / file_part
                if not file_path.is_file():
                    missing.append(nodeid)
                    continue

                functions = function_cache.get(file_path)
                if functions is None:
                    try:
                        tree = ast.parse(
                            file_path.read_text(encoding="utf-8"),
                            filename=str(file_path),
                        )
                    except (OSError, SyntaxError) as exc:
                        raise StatefulFaultSuiteValidationError(
                            f"cannot inspect pytest module: {file_path}"
                        ) from exc
                    functions = {
                        node.name
                        for node in tree.body
                        if isinstance(
                            node,
                            (ast.FunctionDef, ast.AsyncFunctionDef),
                        )
                    }
                    function_cache[file_path] = functions

                function_name = function_part.split("[", 1)[0]
                if function_name not in functions:
                    missing.append(nodeid)

        if missing:
            raise StatefulFaultSuiteValidationError(
                "stateful fault suite references missing pytest nodeids: "
                + ", ".join(sorted(missing))
            )
