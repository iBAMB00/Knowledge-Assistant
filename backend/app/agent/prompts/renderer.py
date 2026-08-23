"""统一 Prompt 渲染，避免 Native / LangChain / LangGraph 各自拼装。"""

from collections.abc import Mapping

from app.agent.prompts.contracts import PromptTemplateContract, RenderedPrompt


class PromptRenderer:
    """按显式变量边界渲染 Prompt，缺失或额外变量均 fail closed。"""

    def render(
        self,
        contract: PromptTemplateContract,
        variables: Mapping[str, object] | None = None,
    ) -> RenderedPrompt:
        values = dict(variables or {})
        expected = set(contract.variables)
        provided = set(values)

        missing = sorted(expected - provided)
        if missing:
            raise ValueError(
                "missing prompt variables: " + ", ".join(missing)
            )

        unexpected = sorted(provided - expected)
        if unexpected:
            raise ValueError(
                "unexpected prompt variables: " + ", ".join(unexpected)
            )

        if contract.variables:
            normalized_values = {
                key: str(value)
                for key, value in values.items()
            }
            content = contract.template.format_map(normalized_values)
        else:
            # 未声明变量时保留字面量大括号，兼容 JSON 示例等 Prompt 内容。
            content = contract.template

        if not content.strip():
            raise ValueError("rendered prompt cannot be empty")

        return RenderedPrompt(
            prompt_id=contract.prompt_id,
            version=contract.version,
            content=content,
        )
