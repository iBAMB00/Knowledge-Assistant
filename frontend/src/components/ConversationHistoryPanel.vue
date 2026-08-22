<script setup lang="ts">
import { Clock3, LoaderCircle, MessageSquareText, Plus, Trash2 } from "lucide-vue-next";
import { computed } from "vue";
import type {
  ChatMode,
  ConversationRecord,
  KnowledgeBaseRecord,
} from "@/types/knowledge";

const props = defineProps<{
  conversations: ConversationRecord[];
  knowledgeBases: KnowledgeBaseRecord[];
  activeConversationId?: number;
  currentMode: ChatMode;
  loading: boolean;
  openingConversationId?: number;
  deletingConversationId?: number;
}>();

const emit = defineEmits<{
  newConversation: [];
  openConversation: [conversation: ConversationRecord];
  deleteConversation: [conversation: ConversationRecord];
}>();

const sortedConversations = computed(() =>
  [...props.conversations].sort(
    (a, b) => Date.parse(b.updated_at) - Date.parse(a.updated_at),
  ),
);

function knowledgeBaseName(knowledgeBaseId: number): string {
  return props.knowledgeBases.find((item) => item.id === knowledgeBaseId)?.name ?? `知识库 #${knowledgeBaseId}`;
}

function modeLabel(conversation: ConversationRecord): string {
  return conversation.mode === "rag" ? "RAG" : "Agent";
}

function timeLabel(value: string): string {
  const timestamp = Date.parse(value);
  if (!Number.isFinite(timestamp)) return "";
  const date = new Date(timestamp);
  const now = new Date();
  const sameDay =
    date.getFullYear() === now.getFullYear() &&
    date.getMonth() === now.getMonth() &&
    date.getDate() === now.getDate();
  if (sameDay) {
    return new Intl.DateTimeFormat("zh-CN", {
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
    }).format(date);
  }
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
  }).format(date);
}

function requestDelete(conversation: ConversationRecord): void {
  emit("deleteConversation", conversation);
}
</script>

<template>
  <aside class="conversation-history-panel">
    <div class="conversation-history-heading">
      <div>
        <strong>历史对话</strong>
        <span>按最近活动排序</span>
      </div>
      <button
        type="button"
        class="icon-button ghost conversation-new-icon"
        title="新建对话"
        @click="emit('newConversation')"
      >
        <Plus :size="16" />
      </button>
    </div>

    <button
      type="button"
      class="conversation-new-button"
      @click="emit('newConversation')"
    >
      <Plus :size="15" />
      新建{{ currentMode === 'knowledge' ? '知识问答' : ' Agent 对话' }}
    </button>

    <div v-if="loading" class="conversation-history-state">
      <LoaderCircle :size="16" class="spinning" />
      <span>加载历史对话…</span>
    </div>

    <div v-else-if="sortedConversations.length === 0" class="conversation-history-empty">
      <MessageSquareText :size="24" />
      <strong>还没有历史对话</strong>
      <span>发送第一条消息后会自动保存。</span>
    </div>

    <div v-else class="conversation-history-list">
      <article
        v-for="conversation in sortedConversations"
        :key="conversation.id"
        class="conversation-history-item"
        :class="{ active: conversation.id === activeConversationId }"
      >
        <button
          type="button"
          class="conversation-history-main"
          :disabled="openingConversationId === conversation.id"
          @click="emit('openConversation', conversation)"
        >
          <span class="conversation-history-title">
            <LoaderCircle
              v-if="openingConversationId === conversation.id"
              :size="12"
              class="spinning"
            />
            <span>{{ conversation.title || '新对话' }}</span>
          </span>
          <span class="conversation-history-meta">
            <span class="conversation-mode-badge" :class="`mode-${conversation.mode}`">
              {{ modeLabel(conversation) }}
            </span>
            <span class="conversation-kb-name">{{ knowledgeBaseName(conversation.knowledge_base_id) }}</span>
          </span>
          <span class="conversation-history-time">
            <Clock3 :size="11" />
            {{ timeLabel(conversation.updated_at) }}
          </span>
        </button>

        <button
          type="button"
          class="icon-button ghost conversation-delete-button"
          :disabled="deletingConversationId === conversation.id"
          title="删除对话"
          @click.stop="requestDelete(conversation)"
        >
          <LoaderCircle
            v-if="deletingConversationId === conversation.id"
            :size="13"
            class="spinning"
          />
          <Trash2 v-else :size="13" />
        </button>
      </article>
    </div>
  </aside>
</template>
