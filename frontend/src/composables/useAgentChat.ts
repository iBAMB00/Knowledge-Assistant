import { nextTick, reactive, ref } from "vue";
import { chatWithAgent, getAgentRuntimes, streamAgentChat } from "@/api/agent";
import type {
  AgentActivityRecord,
  AgentChatRequest,
  AgentRuntime,
  AgentRuntimeCapability,
  AgentToolCallEvent,
  AgentToolResultEvent,
  ChatMessageRecord,
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
      runtimeError.value = toErrorMessage(error);
    } finally {
      runtimeLoading.value = false;
    }
  }

  async function sendQuestion(
    question: string,
    knowledgeBaseId: number,
  ): Promise<void> {
    const normalized = question.trim();
    if (!normalized || submitting.value) return;

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
    };
    const startedAt = performance.now();

    try {
      if (streamingEnabled.value) {
        const controller = new AbortController();
        abortController.value = controller;
        await streamAgentChat(
          payload,
          selectedRuntime.value,
          {
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
            onMessage(content) {
              answer.content = content;
              void notify();
            },
            onDone() {},
          },
          controller.signal,
        );
      } else {
        const response = await chatWithAgent(payload, selectedRuntime.value);
        answer.content = response.answer;
      }
    } catch (error) {
      if (isAbortError(error)) {
        answer.content ||= "Agent 运行已停止。";
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

  function setRuntime(runtime: AgentRuntime): void {
    const option = runtimeOptions.value.find((item) => item.runtime === runtime);
    if (!option?.enabled || submitting.value) return;
    selectedRuntime.value = runtime;
  }

  function stopGeneration(): void {
    abortController.value?.abort();
  }

  function clearConversation(): void {
    stopGeneration();
    messages.value = [
      createAssistantMessage("agent-welcome", welcomeContent, false),
    ];
  }

  function resetRuntimeState(): void {
    selectedRuntime.value = "native";
    runtimeOptions.value = [...fallbackRuntimes];
    runtimeError.value = "";
    runtimesLoaded.value = false;
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
    loadRuntimes,
    setRuntime,
    sendQuestion,
    stopGeneration,
    clearConversation,
    resetRuntimeState,
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

function toolProvider(toolName: string): "local" | "mcp" {
  return toolName.startsWith("mcp__") ? "mcp" : "local";
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
