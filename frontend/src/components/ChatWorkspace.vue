<script setup lang="ts">
import { Bot, RotateCcw, Search } from "lucide-vue-next";
import AgentChatPanel from "@/components/AgentChatPanel.vue";
import KnowledgeChatPanel from "@/components/KnowledgeChatPanel.vue";
import type {
  AgentRuntime,
  AgentRuntimeCapability,
  ChatMessageRecord,
  ChatMode,
  DocumentRecord,
  KnowledgeBaseRecord,
} from "@/types/knowledge";

const props = defineProps<{
  mode: ChatMode;
  knowledgeBases: KnowledgeBaseRecord[];
  selectedKnowledgeBaseId?: number;
  documents: DocumentRecord[];
  selectedDocumentId?: number;
  knowledgeMessages: ChatMessageRecord[];
  knowledgeSubmitting: boolean;
  knowledgeStreamingEnabled: boolean;
  agentMessages: ChatMessageRecord[];
  agentSubmitting: boolean;
  agentStreamingEnabled: boolean;
  agentRuntime: AgentRuntime;
  agentRuntimeOptions: AgentRuntimeCapability[];
  agentRuntimeLoading: boolean;
  agentRuntimeError: string;
}>();

const emit = defineEmits<{
  "update:mode": [value: ChatMode];
  "update:selectedKnowledgeBaseId": [value?: number];
  "update:selectedDocumentId": [value?: number];
  "update:knowledgeStreamingEnabled": [value: boolean];
  "update:agentStreamingEnabled": [value: boolean];
  "update:agentRuntime": [value: AgentRuntime];
  sendKnowledge: [question: string];
  sendAgent: [question: string];
  stopKnowledge: [];
  stopAgent: [];
  clearKnowledge: [];
  clearAgent: [];
}>();

function switchMode(mode: ChatMode): void {
  if (props.mode === mode) return;
  emit("update:mode", mode);
}
</script>

<template>
  <section class="chat-page-shell">
    <header class="chat-heading assistant-heading">
      <div>
        <p class="eyebrow">AI Workspace</p>
        <h1>AI 助手</h1>
        <p>保留稳定 RAG 知识问答，同时提供可观察的 Agent Tool Calling 与 MCP 外部工具能力。</p>
      </div>
      <button type="button" class="secondary-button" @click="mode === 'knowledge' ? emit('clearKnowledge') : emit('clearAgent')"><RotateCcw :size="16" />新对话</button>
    </header>

    <div class="assistant-mode-switch" role="tablist" aria-label="AI 助手模式">
      <button type="button" :class="{ active: mode === 'knowledge' }" role="tab" :aria-selected="mode === 'knowledge'" @click="switchMode('knowledge')">
        <Search :size="15" />
        <span><strong>知识问答</strong><small>稳定 RAG 基线</small></span>
      </button>
      <button type="button" :class="{ active: mode === 'agent' }" role="tab" :aria-selected="mode === 'agent'" @click="switchMode('agent')">
        <Bot :size="16" />
        <span><strong>Agent 助手</strong><small>Tool Calling · MCP</small></span>
      </button>
    </div>

    <KnowledgeChatPanel
      v-if="mode === 'knowledge'"
      :knowledge-bases="knowledgeBases"
      :selected-knowledge-base-id="selectedKnowledgeBaseId"
      :documents="documents"
      :selected-document-id="selectedDocumentId"
      :messages="knowledgeMessages"
      :submitting="knowledgeSubmitting"
      :streaming-enabled="knowledgeStreamingEnabled"
      @update:selected-knowledge-base-id="emit('update:selectedKnowledgeBaseId', $event)"
      @update:selected-document-id="emit('update:selectedDocumentId', $event)"
      @update:streaming-enabled="emit('update:knowledgeStreamingEnabled', $event)"
      @send="emit('sendKnowledge', $event)"
      @stop="emit('stopKnowledge')"
    />

    <AgentChatPanel
      v-else
      :knowledge-bases="knowledgeBases"
      :selected-knowledge-base-id="selectedKnowledgeBaseId"
      :messages="agentMessages"
      :submitting="agentSubmitting"
      :streaming-enabled="agentStreamingEnabled"
      :selected-runtime="agentRuntime"
      :runtime-options="agentRuntimeOptions"
      :runtime-loading="agentRuntimeLoading"
      :runtime-error="agentRuntimeError"
      @update:selected-knowledge-base-id="emit('update:selectedKnowledgeBaseId', $event)"
      @update:streaming-enabled="emit('update:agentStreamingEnabled', $event)"
      @update:selected-runtime="emit('update:agentRuntime', $event)"
      @send="emit('sendAgent', $event)"
      @stop="emit('stopAgent')"
    />
  </section>
</template>
