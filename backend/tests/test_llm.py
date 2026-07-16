from app.services.llm_service import LLMService


def main() -> None:
    llm_service = LLMService()
    answer = llm_service.chat("请用一句话解释什么是网络安全防护覆盖率。")
    print(answer)


if __name__ == "__main__":
    main()