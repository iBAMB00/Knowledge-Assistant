from app.core.config import get_settings
from app.services.reranker.bailian import BailianRerankerProvider
from app.services.reranker.base import RerankerProvider


class RerankerFactory:
    """根据配置创建 Reranker Provider。"""

    @staticmethod
    def create() -> RerankerProvider:
        """创建当前配置的重排序 Provider。"""

        settings = get_settings()
        provider = settings.reranker_provider.strip().lower()

        if provider == "bailian":
            return BailianRerankerProvider(
                api_key=RerankerFactory._require_value(
                    settings.reranker_api_key,
                    "reranker_api_key",
                ),
                base_url=RerankerFactory._require_value(
                    settings.reranker_base_url,
                    "reranker_base_url",
                ),
                model=settings.reranker_model,
                timeout=settings.reranker_timeout,
                instruct=settings.reranker_instruct,
            )

        raise ValueError(
            "unsupported reranker provider: "
            f"{provider}"
        )

    @staticmethod
    def _require_value(value: str | None, field_name: str) -> str:
        """校验必需配置。"""

        if value is None or not value.strip():
            raise ValueError(f"{field_name} is required")
        return value.strip()
