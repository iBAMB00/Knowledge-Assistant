"""当前 OpenAI-compatible 模型配置到 LangChain ChatModel 的薄适配层。"""

from app.core.config import Settings, get_settings


class LangChainModelAdapter:
    """根据项目现有 MODEL_* 配置构建 LangChain ChatModel。

    当前生产 LLMService 本身使用 OpenAI-compatible Chat Completions，
    因此 v2.1-A2 先采用 ChatOpenAI 作为最小 Framework Candidate。
    未来如果 provider 需要非标准能力，应在这里新增 provider-specific
    adapter，而不是把判断扩散到 Agent Runner。
    """

    ADAPTER_VERSION = "1.0.0"

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def build(self):
        """构建与当前 Native LLM 基本参数对齐的 LangChain ChatModel。"""

        ChatOpenAI = self._load_chat_openai()
        return ChatOpenAI(
            model=self._settings.model_name,
            api_key=self._settings.model_api_key,
            base_url=self._settings.model_base_url,
            temperature=0.2,
            timeout=60,
        )

    @staticmethod
    def _load_chat_openai():
        """延迟导入 provider integration，保持 Native 路径无框架硬依赖。"""

        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "LangChain model adapter requires langchain-openai; "
                "install project requirements before using v2.1 framework integration"
            ) from exc

        return ChatOpenAI
