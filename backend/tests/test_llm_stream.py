from app.services.llm_service import LLMService


def main() -> None:
    llm_service = LLMService()

    for content in llm_service.stream_chat(
        "请用一句话解释什么是零信任安全。"
    ):
        print(content, end="", flush=True)

    print()


if __name__ == "__main__":
    main()