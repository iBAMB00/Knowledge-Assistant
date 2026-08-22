import { computed, ref } from "vue";
import {
  createConversation,
  deleteConversation,
  listConversationMessages,
  listConversations,
} from "@/api/conversations";
import type {
  ChatMode,
  ConversationBackendMode,
  ConversationMessageRecord,
  ConversationRecord,
} from "@/types/knowledge";

const ACTIVE_CONVERSATION_PREFIX = "knowledge-assistant-active-conversation";

export function useConversationHistory() {
  const conversations = ref<ConversationRecord[]>([]);
  const activeConversationId = ref<number>();
  const loading = ref(false);
  const openingConversationId = ref<number>();
  const deletingConversationId = ref<number>();

  const activeConversation = computed(() =>
    conversations.value.find((item) => item.id === activeConversationId.value),
  );

  async function load(userId: number): Promise<void> {
    loading.value = true;
    try {
      conversations.value = await listConversations({ limit: 100 });
      const storedId = readStoredConversationId(userId);
      if (storedId && conversations.value.some((item) => item.id === storedId)) {
        activeConversationId.value = storedId;
      } else if (
        activeConversationId.value &&
        !conversations.value.some((item) => item.id === activeConversationId.value)
      ) {
        setActive(undefined, userId);
      }
    } finally {
      loading.value = false;
    }
  }

  async function ensureConversation(
    userId: number,
    mode: ChatMode,
    knowledgeBaseId: number,
  ): Promise<ConversationRecord> {
    const backendMode = toBackendMode(mode);
    const current = activeConversation.value;
    if (
      current &&
      current.mode === backendMode &&
      current.knowledge_base_id === knowledgeBaseId
    ) {
      return current;
    }

    const created = await createConversation({
      mode: backendMode,
      knowledge_base_id: knowledgeBaseId,
    });
    conversations.value = [created, ...conversations.value.filter((x) => x.id !== created.id)];
    setActive(created.id, userId);
    return created;
  }

  async function open(
    userId: number,
    conversation: ConversationRecord,
  ): Promise<ConversationMessageRecord[]> {
    openingConversationId.value = conversation.id;
    try {
      const messages = await listConversationMessages(conversation.id);
      setActive(conversation.id, userId);
      return messages;
    } finally {
      openingConversationId.value = undefined;
    }
  }

  async function remove(userId: number, conversationId: number): Promise<void> {
    deletingConversationId.value = conversationId;
    try {
      await deleteConversation(conversationId);
      conversations.value = conversations.value.filter((item) => item.id !== conversationId);
      if (activeConversationId.value === conversationId) {
        setActive(undefined, userId);
      }
    } finally {
      deletingConversationId.value = undefined;
    }
  }

  async function refresh(userId: number): Promise<void> {
    const currentId = activeConversationId.value;
    conversations.value = await listConversations({ limit: 100 });
    if (currentId && conversations.value.some((item) => item.id === currentId)) {
      setActive(currentId, userId);
    } else if (currentId) {
      setActive(undefined, userId);
    }
  }

  function startDraft(userId: number): void {
    setActive(undefined, userId);
  }

  function reset(userId?: number): void {
    conversations.value = [];
    activeConversationId.value = undefined;
    openingConversationId.value = undefined;
    deletingConversationId.value = undefined;
    if (userId) localStorage.removeItem(storageKey(userId));
  }

  function setActive(conversationId: number | undefined, userId: number): void {
    activeConversationId.value = conversationId;
    if (conversationId) {
      localStorage.setItem(storageKey(userId), String(conversationId));
    } else {
      localStorage.removeItem(storageKey(userId));
    }
  }

  return {
    conversations,
    activeConversationId,
    activeConversation,
    loading,
    openingConversationId,
    deletingConversationId,
    load,
    refresh,
    ensureConversation,
    open,
    remove,
    startDraft,
    reset,
  };
}

export function toBackendMode(mode: ChatMode): ConversationBackendMode {
  return mode === "knowledge" ? "rag" : "agent";
}

export function toChatMode(mode: ConversationBackendMode): ChatMode {
  return mode === "rag" ? "knowledge" : "agent";
}

function storageKey(userId: number): string {
  return `${ACTIVE_CONVERSATION_PREFIX}:${userId}`;
}

function readStoredConversationId(userId: number): number | undefined {
  const raw = localStorage.getItem(storageKey(userId));
  if (!raw) return undefined;
  const parsed = Number(raw);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : undefined;
}
