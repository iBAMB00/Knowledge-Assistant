import axios from "axios";
import { nextTick, reactive, ref } from "vue";
import {
  approveAgentThread,
  cancelAgentThread,
  chatWithAgent,
  getAgentRuntimes,
  getAgentThreadStatus,
  rejectAgentThread,
  resumeAgentThread,
  streamAgentChat,
  streamResumeAgentThread,
} from "@/api/agent";
import { getApiErrorMessage } from "@/api/http";
import type {
  AgentActivityRecord,
  AgentChatRequest,
  AgentRuntime,
  AgentRuntimeCapability,
  AgentStreamCallbacks,
  AgentThreadAction,
  AgentThreadStatusResponse,
  AgentToolCallEvent,
  AgentToolResultEvent,
  AgentWaitingResponse,
  ChatMessageRecord,
  ConversationMessageRecord,
} from "@/types/knowledge";

const welcomeContent =
  "你好，我是 Agent Assistant。我会在当前知识库权限范围内自主选择受控工具，并在运行过程中展示安全的 Tool Calling 事件。";

const fallbackRuntimes: AgentRuntimeCapability[] = [
  {
    runtime: "native",
    role: "baseline",
    enabled: true,
    supports_sync: true,
    supports_stream: true,
    implementation_version: "native",
  },
];

interface ActiveRunContext {
  conversationId: number;
  knowledgeBaseId: number;
  threadId: string;
}

export function useAgentChat(onUpdated?: () => void) {
  const messages = ref<ChatMessageRecord[]>([
    createAssistantMessage("agent-welcome", welcomeContent, false),
  ]);
  const submitting = ref(false);
  const streamingEnabled = ref(true);
  const abortController = ref<AbortController | null>(null);
  const selectedRuntime = ref<AgentRuntime>("native");
  const runtimeOptions = ref<AgentRuntimeCapability[]>([...fallbackRuntimes]);
  const runtimeLoading = ref(false);
  const runtimeError = ref("");
  const runtimesLoaded = ref(false);

  const threadStatus = ref<AgentThreadStatusResponse | null>(null);
  const threadLoading = ref(false);
  const threadActionBusy = ref<AgentThreadAction>();
  const threadError = ref("");
  const activeRunContext = ref<ActiveRunContext | null>(null);

  async function loadRuntimes(force = false): Promise<void> {
    if (runtimeLoading.value || (runtimesLoaded.value && !force)) return;

    runtimeLoading.value = true;
    runtimeError.value = "";
    try {
      const status = await getAgentRuntimes();
      runtimeOptions.value = status.runtimes;
      const selected = status.runtimes.find(
        (runtime) => runtime.runtime === selectedRuntime.value && runtime.enabled,
      );
      if (!selected) {
        const preferred = status.runtimes.find(
          (runtime) => runtime.runtime === status.default_runtime && runtime.enabled,
        );
        const firstEnabled = status.runtimes.find((runtime) => runtime.enabled);
        selectedRuntime.value = preferred?.runtime ?? firstEnabled?.runtime ?? "native";
      }
      runtimesLoaded.value = true;
    } catch (error) {
      runtimeOptions.value = [...fallbackRuntimes];
      selectedRuntime.value = "native";
      runtimeError.value = getApiErrorMessage(error);
    } finally {
      runtimeLoading.value = false;
    }
  }

  async function loadThreadStatus(
    conversationId: number,
    knowledgeBaseId: number,
    quietNotFound = true,
  ): Promise<void> {
    const threadId = threadIdForConversation(conversationId);
    threadLoading.value = true;
    threadError.value = "";
    try {
      const snapshot = await getAgentThreadStatus(threadId, knowledgeBaseId);
      threadStatus.value = snapshot;
      const langgraph = runtimeOptions.value.find(
        (item) => item.runtime === "langgraph" && item.enabled,
      );
      if (langgraph) selectedRuntime.value = "langgraph";
    } catch (error) {
      if (quietNotFound && isHttpStatus(error, 404)) {
        threadStatus.value = null;
        selectedRuntime.value = "native";
        return;
      }
      threadError.value = getApiErrorMessage(error);
      throw error;
    } finally {
      threadLoading.value = false;
      await notify();
    }
  }

  async function sendQuestion(
    question: string,
    knowledgeBaseId: number,
    conversationId?: number,
  ): Promise<void> {
    const normalized = question.trim();
    if (!normalized || submitting.value) return;

    if (selectedRuntime.value === "langgraph" && !conversationId) {
      throw new Error("LangGraph Stateful Runtime 需要先创建对话。");
    }

    messages.value.push({
      id: createId("agent-user"),
      role: "user",
      content: normalized,
      sources: [],
      createdAt: new Date(),
    });

    const answer = reactive(
      createAssistantMessage(createId("agent-assistant"), "", true),
    );
    answer.agentActivities = [];
    messages.value.push(answer);
    submitting.value = true;
    await notify();

    const payload: AgentChatRequest = {
      message: normalized,
      knowledge_base_id: knowledgeBaseId,
      ...(conversationId ? { conversation_id: conversationId } : {}),
    };
    const startedAt = performance.now();
    const isLangGraph = selectedRuntime.value === "langgraph";

    if (isLangGraph && conversationId) {
      const context = {
        conversationId,
        knowledgeBaseId,
        threadId: threadIdForConversation(conversationId),
      };
      activeRunContext.value = context;
      threadStatus.value = provisionalRunningThread(context, threadStatus.value);
      threadError.value = "";
    }

    try {
      if (streamingEnabled.value) {
        const controller = new AbortController();
        abortController.value = controller;
        await streamAgentChat(
          payload,
          selectedRuntime.value,
          createStreamCallbacks(answer, knowledgeBaseId, conversationId),
          controller.signal,
        );
      } else {
        const response = await chatWithAgent(payload, selectedRuntime.value);
        if (isWaitingResponse(response)) {
          applyWaitingResponse(answer, response, knowledgeBaseId, conversationId);
        } else {
          answer.content = response.answer;
          answer.contextUsage = response.context_usage ?? undefined;
        }
      }
    } catch (error) {
      if (isAbortError(error)) {
        answer.content ||= isLangGraph
          ? "Agent 运行已停止，正在同步 Stateful Thread 状态。"
          : "Agent 运行已停止。";
      } else {
        answer.error = true;
        answer.content = getApiErrorMessage(error);
      }
    } finally {
      answer.pending = false;
      answer.elapsedMs = Math.round(performance.now() - startedAt);
      submitting.value = false;
      abortController.value = null;
      activeRunContext.value = null;
      if (isLangGraph && conversationId) {
        await loadThreadStatus(conversationId, knowledgeBaseId, true);
      }
      await notify();
    }
  }

  async function approveThread(knowledgeBaseId: number): Promise<void> {
    const current = requireThreadStatus();
    await runThreadAction("approve", async () => {
      threadStatus.value = await approveAgentThread(
        current.thread_id,
        knowledgeBaseId,
        current.pending_approvals.map((item) => item.call_id),
      );
    });
  }

  async function rejectThread(knowledgeBaseId: number): Promise<void> {
    const current = requireThreadStatus();
    await runThreadAction("reject", async () => {
      threadStatus.value = await rejectAgentThread(
        current.thread_id,
        knowledgeBaseId,
        current.pending_approvals.map((item) => item.call_id),
      );
    });
  }

  async function cancelThread(knowledgeBaseId: number): Promise<void> {
    const current = requireThreadStatus();
    await runThreadAction("cancel", async () => {
      threadStatus.value = await cancelAgentThread(
        current.thread_id,
        knowledgeBaseId,
      );
      abortController.value?.abort();
    });
  }

  async function resumeThread(knowledgeBaseId: number): Promise<void> {
    const current = requireThreadStatus();
    if (!current.can_resume || submitting.value) return;

    const answer = reactive(
      createAssistantMessage(createId("agent-resume"), "", true),
    );
    answer.agentActivities = [];
    messages.value.push(answer);
    submitting.value = true;
    threadActionBusy.value = "resume";
    threadError.value = "";
    activeRunContext.value = {
      conversationId: current.conversation_id,
      knowledgeBaseId,
      threadId: current.thread_id,
    };
    const startedAt = performance.now();
    await notify();

    try {
      if (streamingEnabled.value) {
        const controller = new AbortController();
        abortController.value = controller;
        await streamResumeAgentThread(
          current.thread_id,
          knowledgeBaseId,
          createStreamCallbacks(answer, knowledgeBaseId, current.conversation_id),
          controller.signal,
        );
      } else {
        const response = await resumeAgentThread(current.thread_id, knowledgeBaseId);
        if (isWaitingResponse(response)) {
          applyWaitingResponse(
            answer,
            response,
            knowledgeBaseId,
            current.conversation_id,
          );
        } else {
          answer.content = response.answer;
          answer.contextUsage = response.context_usage ?? undefined;
        }
      }
    } catch (error) {
      if (isAbortError(error)) {
        answer.content ||= "Agent 恢复执行已停止，正在同步 Thread 状态。";
      } else {
        answer.error = true;
        answer.content = getApiErrorMessage(error);
        threadError.value = answer.content;
      }
    } finally {
      answer.pending = false;
      answer.elapsedMs = Math.round(performance.now() - startedAt);
      submitting.value = false;
      abortController.value = null;
      activeRunContext.value = null;
      threadActionBusy.value = undefined;
      await loadThreadStatus(current.conversation_id, knowledgeBaseId, true);
      await notify();
    }
  }

  async function refreshThreadStatus(): Promise<void> {
    const current = requireThreadStatus();
    threadActionBusy.value = "refresh";
    try {
      await loadThreadStatus(current.conversation_id, current.knowledge_base_id, false);
    } finally {
      threadActionBusy.value = undefined;
    }
  }

  async function stopGeneration(): Promise<void> {
    const controller = abortController.value;
    const context = activeRunContext.value;

    if (selectedRuntime.value === "langgraph" && context) {
      try {
        threadActionBusy.value = "cancel";
        threadStatus.value = await cancelAgentThread(
          context.threadId,
          context.knowledgeBaseId,
        );
      } catch (error) {
        // Thread 可能尚未来得及创建 durable checkpoint；仍然停止当前 SSE。
        if (!isHttpStatus(error, 404)) {
          threadError.value = getApiErrorMessage(error);
        }
      } finally {
        threadActionBusy.value = undefined;
        controller?.abort();
      }
      return;
    }

    controller?.abort();
  }

  function setRuntime(runtime: AgentRuntime): void {
    const option = runtimeOptions.value.find((item) => item.runtime === runtime);
    if (!option?.enabled || submitting.value) return;
    selectedRuntime.value = runtime;
  }

  function clearConversation(): void {
    void stopGeneration();
    messages.value = [
      createAssistantMessage("agent-welcome", welcomeContent, false),
    ];
    clearThreadState();
  }

  function restoreConversation(history: ConversationMessageRecord[]): void {
    void stopGeneration();
    messages.value = history.length > 0
      ? history.map(toChatMessage)
      : [createAssistantMessage("agent-welcome", welcomeContent, false)];
    clearThreadState();
  }

  function resetRuntimeState(): void {
    selectedRuntime.value = "native";
    runtimeOptions.value = [...fallbackRuntimes];
    runtimeError.value = "";
    runtimesLoaded.value = false;
    clearThreadState();
  }

  function clearThreadState(): void {
    threadStatus.value = null;
    threadLoading.value = false;
    threadActionBusy.value = undefined;
    threadError.value = "";
    activeRunContext.value = null;
  }

  function createStreamCallbacks(
    answer: ChatMessageRecord,
    knowledgeBaseId: number,
    conversationId?: number,
  ): AgentStreamCallbacks {
    return {
      onStatus(event) {
        answer.agentActivities?.push({
          id: createId(`status-${event.turn}`),
          kind: "status",
          turn: event.turn,
        });
        void notify();
      },
      onToolCall(event) {
        upsertToolActivity(answer, event);
        void notify();
      },
      onToolResult(event) {
        finishToolActivity(answer, event);
        void notify();
      },
      onMessage(content, contextUsage) {
        answer.content = content;
        answer.contextUsage = contextUsage;
        void notify();
      },
      onWaiting(event) {
        applyWaitingResponse(answer, event, knowledgeBaseId, conversationId);
        void notify();
      },
      onCancelled(message) {
        answer.content ||= message;
        if (threadStatus.value) {
          threadStatus.value = {
            ...threadStatus.value,
            status: "cancelled",
            can_resume: false,
            can_approve: false,
            can_reject: false,
            can_cancel: false,
          };
        }
        void notify();
      },
      onDone() {},
    };
  }

  function applyWaitingResponse(
    answer: ChatMessageRecord,
    response: AgentWaitingResponse,
    knowledgeBaseId: number,
    conversationId?: number,
  ): void {
    answer.content ||= "Agent 已暂停，等待人工确认后再继续执行。";
    if (!conversationId) return;
    threadStatus.value = {
      thread_id: response.thread_id,
      conversation_id: conversationId,
      knowledge_base_id: knowledgeBaseId,
      status: "waiting",
      retry_count: threadStatus.value?.retry_count ?? 0,
      last_error_code: null,
      pending_approvals: response.approvals,
      can_resume: false,
      can_approve: response.approvals.length > 0,
      can_reject: response.approvals.length > 0,
      can_cancel: true,
    };
  }

  async function runThreadAction(
    action: AgentThreadAction,
    operation: () => Promise<void>,
  ): Promise<void> {
    if (threadActionBusy.value || submitting.value) return;
    threadActionBusy.value = action;
    threadError.value = "";
    try {
      await operation();
    } catch (error) {
      threadError.value = getApiErrorMessage(error);
      throw error;
    } finally {
      threadActionBusy.value = undefined;
      await notify();
    }
  }

  function requireThreadStatus(): AgentThreadStatusResponse {
    if (!threadStatus.value) {
      throw new Error("当前对话还没有 Stateful Agent Thread。");
    }
    return threadStatus.value;
  }

  async function notify(): Promise<void> {
    await nextTick();
    onUpdated?.();
  }

  return {
    messages,
    submitting,
    streamingEnabled,
    selectedRuntime,
    runtimeOptions,
    runtimeLoading,
    runtimeError,
    threadStatus,
    threadLoading,
    threadActionBusy,
    threadError,
    loadRuntimes,
    loadThreadStatus,
    refreshThreadStatus,
    approveThread,
    rejectThread,
    cancelThread,
    resumeThread,
    setRuntime,
    sendQuestion,
    stopGeneration,
    clearConversation,
    restoreConversation,
    resetRuntimeState,
    clearThreadState,
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

function upsertToolActivity(
  answer: ChatMessageRecord,
  event: AgentToolCallEvent,
): void {
  const activities = ensureActivities(answer);
  const existing = activities.find(
    (activity) => activity.kind === "tool" && activity.callId === event.call_id,
  );
  if (existing?.kind === "tool") {
    existing.status = "running";
    existing.turn = event.turn;
    existing.toolName = event.tool_name;
    existing.provider = toolProvider(event.tool_name);
    return;
  }

  activities.push({
    id: event.call_id,
    kind: "tool",
    turn: event.turn,
    callId: event.call_id,
    toolName: event.tool_name,
    provider: toolProvider(event.tool_name),
    status: "running",
    errorCode: null,
  });
}

function finishToolActivity(
  answer: ChatMessageRecord,
  event: AgentToolResultEvent,
): void {
  const activities = ensureActivities(answer);
  const existing = activities.find(
    (activity) => activity.kind === "tool" && activity.callId === event.call_id,
  );

  if (existing?.kind === "tool") {
    existing.status = event.ok ? "succeeded" : "failed";
    existing.errorCode = event.error_code;
    existing.turn = event.turn;
    existing.toolName = event.tool_name;
    existing.provider = toolProvider(event.tool_name);
    return;
  }

  activities.push({
    id: event.call_id,
    kind: "tool",
    turn: event.turn,
    callId: event.call_id,
    toolName: event.tool_name,
    provider: toolProvider(event.tool_name),
    status: event.ok ? "succeeded" : "failed",
    errorCode: event.error_code,
  });
}

function ensureActivities(answer: ChatMessageRecord): AgentActivityRecord[] {
  answer.agentActivities ??= [];
  return answer.agentActivities;
}

function provisionalRunningThread(
  context: ActiveRunContext,
  current: AgentThreadStatusResponse | null,
): AgentThreadStatusResponse {
  return {
    thread_id: context.threadId,
    conversation_id: context.conversationId,
    knowledge_base_id: context.knowledgeBaseId,
    status: "running",
    retry_count: current?.retry_count ?? 0,
    last_error_code: null,
    pending_approvals: [],
    can_resume: false,
    can_approve: false,
    can_reject: false,
    can_cancel: true,
  };
}

function threadIdForConversation(conversationId: number): string {
  return `conversation:${conversationId}`;
}

function isWaitingResponse(
  response: { answer: string } | AgentWaitingResponse,
): response is AgentWaitingResponse {
  return "status" in response && response.status === "waiting";
}

function toolProvider(toolName: string): "local" | "mcp" {
  return toolName.startsWith("mcp__") ? "mcp" : "local";
}

function createId(prefix: string): string {
  return `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

function isHttpStatus(error: unknown, status: number): boolean {
  return axios.isAxiosError(error) && error.response?.status === status;
}

function toChatMessage(message: ConversationMessageRecord): ChatMessageRecord {
  return {
    id: `agent-history-${message.id}`,
    role: message.role,
    content: message.content,
    sources: [],
    createdAt: new Date(message.created_at),
  };
}
