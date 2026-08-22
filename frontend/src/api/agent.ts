import { getAccessToken, http } from "@/api/http";
import type {
  AgentChatRequest,
  AgentChatResponse,
  AgentRuntime,
  AgentRuntimeStatusResponse,
  AgentStreamCallbacks,
} from "@/types/knowledge";
import { consumeSse } from "@/utils/sse";

export async function getAgentRuntimes(): Promise<AgentRuntimeStatusResponse> {
  const response = await http.get<AgentRuntimeStatusResponse>("/agent/runtimes");
  return response.data;
}

export async function chatWithAgent(
  payload: AgentChatRequest,
  runtime: AgentRuntime,
): Promise<AgentChatResponse> {
  const response = await http.post<AgentChatResponse>("/agent/chat", payload, {
    params: { runtime },
  });
  return response.data;
}

export async function streamAgentChat(
  payload: AgentChatRequest,
  runtime: AgentRuntime,
  callbacks: AgentStreamCallbacks,
  signal?: AbortSignal,
): Promise<void> {
  const token = getAccessToken();
  const query = new URLSearchParams({ runtime });
  const response = await fetch(`/agent/chat/stream?${query.toString()}`, {
    method: "POST",
    headers: {
      Accept: "text/event-stream",
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify(payload),
    signal,
  });

  if (!response.ok) {
    if (response.status === 401) {
      window.dispatchEvent(new CustomEvent("knowledge-assistant:unauthorized"));
    }
    throw new Error(await readResponseError(response));
  }

  await consumeSse(response, ({ event, data }) => {
    const parsed = parseJson(data);

    if (event === "status") {
      callbacks.onStatus({
        turn: toNumber(parsed?.turn) ?? 1,
        stage: parsed?.stage === "model" ? "model" : "model",
      });
      return;
    }

    if (event === "tool_call") {
      const callId = toStringValue(parsed?.call_id);
      const toolName = toStringValue(parsed?.tool_name);
      if (!callId || !toolName) return;
      callbacks.onToolCall({
        turn: toNumber(parsed?.turn) ?? 1,
        call_id: callId,
        tool_name: toolName,
      });
      return;
    }

    if (event === "tool_result") {
      const callId = toStringValue(parsed?.call_id);
      const toolName = toStringValue(parsed?.tool_name);
      if (!callId || !toolName) return;
      callbacks.onToolResult({
        turn: toNumber(parsed?.turn) ?? 1,
        call_id: callId,
        tool_name: toolName,
        ok: parsed?.ok === true,
        error_code: toStringValue(parsed?.error_code) ?? null,
      });
      return;
    }

    if (event === "message") {
      const content = toStringValue(parsed?.content);
      if (content) callbacks.onMessage(content);
      return;
    }

    if (event === "done") {
      callbacks.onDone();
      return;
    }

    if (event === "error") {
      throw new Error(
        toStringValue(parsed?.message) ?? "Agent 问答失败",
      );
    }
  });
}

function parseJson(value: string): Record<string, unknown> | null {
  try {
    const result: unknown = JSON.parse(value);
    return result && typeof result === "object" ? (result as Record<string, unknown>) : null;
  } catch {
    return null;
  }
}

async function readResponseError(response: Response): Promise<string> {
  const text = await response.text();
  try {
    const parsed = JSON.parse(text) as { detail?: unknown };
    if (typeof parsed.detail === "string") return parsed.detail;
  } catch {
    // 使用原始文本。
  }
  return text || `请求失败（HTTP ${response.status}）`;
}

function toNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function toStringValue(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value : undefined;
}
