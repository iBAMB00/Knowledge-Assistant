<script setup lang="ts">
import { Database, RotateCcw, Send, Square } from "lucide-vue-next";
import { nextTick, ref, watch } from "vue";
import ChatMessage from "@/components/ChatMessage.vue";
import type { ChatMessageRecord, DocumentRecord, KnowledgeBaseRecord } from "@/types/knowledge";

const props = defineProps<{
  knowledgeBases: KnowledgeBaseRecord[];
  selectedKnowledgeBaseId?: number;
  documents: DocumentRecord[];
  selectedDocumentId?: number;
  messages: ChatMessageRecord[];
  submitting: boolean;
  streamingEnabled: boolean;
}>();

const emit = defineEmits<{
  "update:selectedKnowledgeBaseId": [value?: number];
  "update:selectedDocumentId": [value?: number];
  "update:streamingEnabled": [value: boolean];
  send: [question: string];
  stop: [];
  clear: [];
}>();

const question = ref("");
const viewport = ref<HTMLElement | null>(null);

watch(() => props.messages.map((message) => `${message.id}:${message.content.length}`).join("|"), async () => {
  await nextTick();
  viewport.value?.scrollTo({ top: viewport.value.scrollHeight, behavior: "smooth" });
});

function submit(): void {
  const normalized = question.value.trim();
  if (!normalized || props.submitting || !props.selectedKnowledgeBaseId) return;
  emit("send", normalized);
  question.value = "";
}

function onKeydown(event: KeyboardEvent): void {
  if (event.key !== "Enter" || event.shiftKey) return;
  event.preventDefault();
  submit();
}

function onKbChange(event: Event): void {
  const value = (event.target as HTMLSelectElement).value;
  emit("update:selectedKnowledgeBaseId", value ? Number(value) : undefined);
}

function onDocChange(event: Event): void {
  const value = (event.target as HTMLSelectElement).value;
  emit("update:selectedDocumentId", value ? Number(value) : undefined);
}

function onStreamingChange(event: Event): void {
  emit(
    "update:streamingEnabled",
    (event.target as HTMLInputElement).checked,
  );
}
</script>

<template>
  <section class="chat-page-shell">
    <header class="chat-heading">
      <div>
        <p class="eyebrow">Knowledge Chat</p>
        <h1>聊天问答</h1>
        <p>基于您有权访问的知识库进行检索增强问答。</p>
      </div>
      <button type="button" class="secondary-button" @click="emit('clear')"><RotateCcw :size="16" />新对话</button>
    </header>

    <div class="chat-layout">
      <aside class="chat-context-panel">
        <label class="form-field compact-field">
          <span>选择知识库</span>
          <div class="select-wrap"><Database :size="15" /><select :value="selectedKnowledgeBaseId ?? ''" @change="onKbChange"><option value="">请选择知识库</option><option v-for="kb in knowledgeBases" :key="kb.id" :value="kb.id">{{ kb.name }}</option></select></div>
        </label>

        <label class="form-field compact-field">
          <span>限定文档（可选）</span>
          <select class="plain-select" :value="selectedDocumentId ?? ''" :disabled="!selectedKnowledgeBaseId" @change="onDocChange">
            <option value="">整个知识库</option>
            <option v-for="document in documents" :key="document.id" :value="document.id">{{ document.filename }}</option>
          </select>
        </label>

        <label class="stream-setting">
          <div><strong>流式回答</strong><span>使用 SSE 实时显示模型输出</span></div>
          <input type="checkbox" :checked="streamingEnabled" @change="onStreamingChange" />
        </label>

        <div class="context-note">
          <strong>检索边界</strong>
          <p>请求会携带当前知识库 ID；后端 RBAC 与检索过滤共同限制可访问范围。</p>
        </div>
      </aside>

      <main class="chat-surface">
        <div ref="viewport" class="messages">
          <ChatMessage v-for="message in messages" :key="message.id" :message="message" />
        </div>

        <div class="chat-composer-wrap">
          <textarea v-model="question" rows="3" maxlength="2000" :disabled="submitting || !selectedKnowledgeBaseId" :placeholder="selectedKnowledgeBaseId ? '输入问题，Enter 发送，Shift + Enter 换行' : '请先选择一个知识库'" @keydown="onKeydown" />
          <div class="chat-composer-footer">
            <span>{{ question.length }} / 2000</span>
            <button v-if="submitting" type="button" class="secondary-button" @click="emit('stop')"><Square :size="14" />停止</button>
            <button v-else type="button" class="primary-button" :disabled="!question.trim() || !selectedKnowledgeBaseId" @click="submit"><Send :size="16" />发送</button>
          </div>
        </div>
      </main>
    </div>
  </section>
</template>
