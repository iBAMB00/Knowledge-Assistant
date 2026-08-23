<script setup lang="ts">
import { Bot, Database, Send, Square } from "lucide-vue-next";
import { computed, nextTick, ref, watch } from "vue";
import AgentThreadControl from "@/components/AgentThreadControl.vue";
import ChatMessage from "@/components/ChatMessage.vue";
import type {
  AgentRuntime,
  AgentRuntimeCapability,
  AgentThreadAction,
  AgentThreadStatusResponse,
  ChatMessageRecord,
  KnowledgeBaseRecord,
} from "@/types/knowledge";

const props = defineProps<{
  knowledgeBases: KnowledgeBaseRecord[];
  selectedKnowledgeBaseId?: number;
  messages: ChatMessageRecord[];
  submitting: boolean;
  streamingEnabled: boolean;
  selectedRuntime: AgentRuntime;
  runtimeOptions: AgentRuntimeCapability[];
  runtimeLoading: boolean;
  runtimeError: string;
  threadStatus?: AgentThreadStatusResponse | null;
  threadLoading: boolean;
  threadActionBusy?: AgentThreadAction;
  threadError: string;
}>();

const emit = defineEmits<{
  "update:selectedKnowledgeBaseId": [value?: number];
  "update:streamingEnabled": [value: boolean];
  "update:selectedRuntime": [value: AgentRuntime];
  send: [question: string];
  stop: [];
  refreshThread: [];
  approveThread: [];
  rejectThread: [];
  resumeThread: [];
  cancelThread: [];
}>();

const question = ref("");
const viewport = ref<HTMLElement | null>(null);

const selectedRuntimeInfo = computed(() =>
  props.runtimeOptions.find((runtime) => runtime.runtime === props.selectedRuntime),
);

const runtimeDisplayName = computed(() => runtimeLabel(props.selectedRuntime));

watch(
  () => props.messages
    .map((message) => `${message.id}:${message.content.length}:${message.agentActivities?.length ?? 0}`)
    .join("|"),
  async () => {
    await nextTick();
    viewport.value?.scrollTo({ top: viewport.value.scrollHeight, behavior: "smooth" });
  },
);

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

function onRuntimeChange(event: Event): void {
  emit("update:selectedRuntime", (event.target as HTMLSelectElement).value as AgentRuntime);
}

function onStreamingChange(event: Event): void {
  emit("update:streamingEnabled", (event.target as HTMLInputElement).checked);
}

function runtimeLabel(runtime: AgentRuntime): string {
  if (runtime === "langchain") return "LangChain";
  if (runtime === "langgraph") return "LangGraph";
  return "Native";
}
</script>

<template>
  <div class="chat-layout agent-chat-layout">
    <aside class="chat-context-panel">
      <div class="mode-context-heading agent-context-heading">
        <strong><Bot :size="14" /> Agent 助手</strong>
        <span>Tool Calling · MCP · Stateful Runtime</span>
      </div>

      <label class="form-field compact-field">
        <span>可信知识范围</span>
        <div class="select-wrap">
          <Database :size="15" />
          <select :value="selectedKnowledgeBaseId ?? ''" @change="onKbChange">
            <option value="">请选择知识库</option>
            <option v-for="kb in knowledgeBases" :key="kb.id" :value="kb.id">{{ kb.name }}</option>
          </select>
        </div>
      </label>

      <label class="form-field compact-field">
        <span>Agent Runtime</span>
        <select
          class="plain-select"
          :value="selectedRuntime"
          :disabled="runtimeLoading || submitting || threadStatus?.status === 'waiting' || threadStatus?.status === 'running'"
          @change="onRuntimeChange"
        >
          <option
            v-for="runtime in runtimeOptions"
            :key="runtime.runtime"
            :value="runtime.runtime"
            :disabled="!runtime.enabled"
          >
            {{ runtimeLabel(runtime.runtime) }} · {{ runtime.role === 'baseline' ? 'Baseline' : 'Candidate' }}{{ runtime.enabled ? '' : '（未启用）' }}
          </option>
        </select>
        <small v-if="selectedRuntimeInfo" class="runtime-version">{{ selectedRuntimeInfo.implementation_version }}</small>
        <small v-if="runtimeError" class="runtime-warning">诊断接口暂不可用，已回退 Native。</small>
      </label>

      <label class="stream-setting">
        <div><strong>运行事件</strong><span>通过 SSE 展示 status / tool_call / tool_result</span></div>
        <input type="checkbox" :checked="streamingEnabled" @change="onStreamingChange" />
      </label>

      <AgentThreadControl
        v-if="selectedRuntime === 'langgraph' || threadStatus || threadLoading || threadError"
        :thread="threadStatus"
        :loading="threadLoading"
        :busy-action="threadActionBusy"
        :error="threadError"
        :submitting="submitting"
        @refresh="emit('refreshThread')"
        @approve="emit('approveThread')"
        @reject="emit('rejectThread')"
        @resume="emit('resumeThread')"
        @cancel="emit('cancelThread')"
      />

      <div class="context-note agent-note">
        <strong>Tool Core</strong>
        <p>Native、LangChain 与 LangGraph 共享 ToolDispatcher、可信上下文和 MCP Tool；LangGraph 额外提供 Checkpoint、Resume、HITL 与 Cancel。</p>
      </div>
    </aside>

    <main class="chat-surface">
      <div ref="viewport" class="messages">
        <ChatMessage v-for="message in messages" :key="message.id" :message="message" />
      </div>

      <div class="chat-composer-wrap agent-composer">
        <textarea
          v-model="question"
          rows="3"
          maxlength="2000"
          :disabled="submitting || !selectedKnowledgeBaseId || threadStatus?.status === 'waiting' || threadStatus?.status === 'running'"
          :placeholder="selectedKnowledgeBaseId ? (threadStatus?.status === 'waiting' ? '当前任务正在等待人工确认，请先处理左侧 Stateful Thread。' : '输入任务，Agent 会自主判断是否调用本地或 MCP 工具') : '请先选择一个知识库作为可信执行范围'"
          @keydown="onKeydown"
        />
        <div class="chat-composer-footer">
          <span>{{ question.length }} / 2000 · {{ runtimeDisplayName }}</span>
          <button v-if="submitting" type="button" class="secondary-button" @click="emit('stop')">
            <Square :size="14" />{{ selectedRuntime === 'langgraph' ? '取消任务' : '停止' }}
          </button>
          <button
            v-else
            type="button"
            class="primary-button"
            :disabled="!question.trim() || !selectedKnowledgeBaseId || threadStatus?.status === 'waiting' || threadStatus?.status === 'running'"
            @click="submit"
          >
            <Send :size="16" />运行 Agent
          </button>
        </div>
      </div>
    </main>
  </div>
</template>
