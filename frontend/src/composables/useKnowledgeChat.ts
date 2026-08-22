import { nextTick, reactive, ref } from "vue";
import { chatWithKnowledge, streamKnowledgeChat } from "@/api/knowledge";
import type {
  ChatMessageRecord,
  ConversationMessageRecord,
  KnowledgeChatRequest,
} from "@/types/knowledge";

const welcomeContent =
  "你好，我是 Knowledge Assistant。请选择一个知识库并开始提问；回答会尽量引用可追溯的知识来源。";

export function useKnowledgeChat(onUpdated?: () => void) {
  const messages = ref<ChatMessageRecord[]>([
    createAssistantMessage("welcome", welcomeContent, false),
  ]);
  const submitting = ref(false);
  const streamingEnabled = ref(true);
  const abortController = ref<AbortController | null>(null);

  async function sendQuestion(
    question: string,
    knowledgeBaseId: number,
    documentId?: number,
    conversationId?: number,
  ): Promise<void> {
    const normalized = question.trim();
    if (!normalized || submitting.value) return;

    messages.value.push({
      id: createId("user"),
      role: "user",
      content: normalized,
      sources: [],
      createdAt: new Date(),
    });

    const answer = reactive(
      createAssistantMessage(createId("assistant"), "", true),
    );
    messages.value.push(answer);
    submitting.value = true;
    await notify();

    const payload: KnowledgeChatRequest = {
      question: normalized,
      knowledge_base_id: knowledgeBaseId,
      top_k: 5,
      ...(conversationId ? { conversation_id: conversationId } : {}),
      ...(documentId ? { document_id: documentId } : {}),
    };

    const startedAt = performance.now();

    try {
      if (streamingEnabled.value) {
        const controller = new AbortController();
        abortController.value = controller;
        await streamKnowledgeChat(
          payload,
          {
            onSources(sources) {
              answer.sources = sources;
              void notify();
            },
            onContent(content) {
              answer.content += content;
              void notify();
            },
            onDone() {},
          },
          controller.signal,
        );
      } else {
        const response = await chatWithKnowledge(payload);
        answer.content = response.answer;
        answer.sources = response.sources;
      }
    } catch (error) {
      if (isAbortError(error)) {
        answer.content ||= "回答已停止。";
      } else {
        answer.error = true;
        answer.content = toErrorMessage(error);
      }
    } finally {
      answer.pending = false;
      answer.elapsedMs = Math.round(performance.now() - startedAt);
      submitting.value = false;
      abortController.value = null;
      await notify();
    }
  }

  function stopGeneration(): void {
    abortController.value?.abort();
  }

  function clearConversation(): void {
    stopGeneration();
    messages.value = [
      createAssistantMessage("welcome", welcomeContent, false),
    ];
  }

  function restoreConversation(history: ConversationMessageRecord[]): void {
    stopGeneration();
    messages.value = history.length > 0
      ? history.map(toChatMessage)
      : [createAssistantMessage("welcome", welcomeContent, false)];
  }

  async function notify(): Promise<void> {
    await nextTick();
    onUpdated?.();
  }

  return {
    messages,
    submitting,
    streamingEnabled,
    sendQuestion,
    stopGeneration,
    clearConversation,
    restoreConversation,
  };
}

function createAssistantMessage(
  id: string,
  content: string,
  pending: boolean,
): ChatMessageRecord {
  return {
    id,
    role: "assistant",
    content,
    sources: [],
    createdAt: new Date(),
    pending,
  };
}

function createId(prefix: string): string {
  return `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

function toErrorMessage(error: unknown): string {
  return error instanceof Error && error.message
    ? error.message
    : "请求失败，请检查后端服务后重试。";
}

function toChatMessage(message: ConversationMessageRecord): ChatMessageRecord {
  return {
    id: `history-${message.id}`,
    role: message.role,
    content: message.content,
    sources: [],
    createdAt: new Date(message.created_at),
  };
}
