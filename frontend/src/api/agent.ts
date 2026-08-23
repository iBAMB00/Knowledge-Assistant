import { getAccessToken, http } from "@/api/http";
import type {
  AgentChatRequest,
  AgentChatResponse,
  AgentRuntime,
  AgentRuntimeStatusResponse,
  AgentStreamCallbacks,
  AgentThreadStatusResponse,
  AgentWaitingResponse,
} from "@/types/knowledge";
import { consumeSse } from "@/utils/sse";

export async function getAgentRuntimes(): Promise<AgentRuntimeStatusResponse> {
  const response = await http.get<AgentRuntimeStatusResponse>("/agent/runtimes");
  return response.data;
}

export async function chatWithAgent(
  payload: AgentChatRequest,
  runtime: AgentRuntime,
): Promise<AgentChatResponse | AgentWaitingResponse> {
  const response = await http.post<AgentChatResponse | AgentWaitingResponse>(
    "/agent/chat",
    payload,
    { params: { runtime } },
  );
  return response.data;
}

export async function streamAgentChat(
  payload: AgentChatRequest,
  runtime: AgentRuntime,
  callbacks: AgentStreamCallbacks,
  signal?: AbortSignal,
): Promise<void> {
  const query = new URLSearchParams({ runtime });
  await streamAgentEndpoint(
    `/agent/chat/stream?${query.toString()}`,
    payload,
    callbacks,
    signal,
  );
}

export async function getAgentThreadStatus(
  threadId: string,
  knowledgeBaseId: number,
): Promise<AgentThreadStatusResponse> {
  const response = await http.get<AgentThreadStatusResponse>(
    `/agent/threads/${encodeURIComponent(threadId)}`,
    { params: { knowledge_base_id: knowledgeBaseId } },
  );
  return response.data;
}

export async function approveAgentThread(
  threadId: string,
  knowledgeBaseId: number,
  callIds: string[] = [],
): Promise<AgentThreadStatusResponse> {
  const response = await http.post<AgentThreadStatusResponse>(
    `/agent/threads/${encodeURIComponent(threadId)}/approve`,
    {
      knowledge_base_id: knowledgeBaseId,
      call_ids: callIds,
    },
  );
  return response.data;
}

export async function rejectAgentThread(
  threadId: string,
  knowledgeBaseId: number,
  callIds: string[] = [],
): Promise<AgentThreadStatusResponse> {
  const response = await http.post<AgentThreadStatusResponse>(
    `/agent/threads/${encodeURIComponent(threadId)}/reject`,
    {
      knowledge_base_id: knowledgeBaseId,
      call_ids: callIds,
    },
  );
  return response.data;
}

export async function cancelAgentThread(
  threadId: string,
  knowledgeBaseId: number,
): Promise<AgentThreadStatusResponse> {
  const response = await http.post<AgentThreadStatusResponse>(
    `/agent/threads/${encodeURIComponent(threadId)}/cancel`,
    { knowledge_base_id: knowledgeBaseId },
  );
  return response.data;
}

export async function resumeAgentThread(
  threadId: string,
  knowledgeBaseId: number,
): Promise<AgentChatResponse | AgentWaitingResponse> {
  const response = await http.post<AgentChatResponse | AgentWaitingResponse>(
    `/agent/threads/${encodeURIComponent(threadId)}/resume`,
    { knowledge_base_id: knowledgeBaseId },
  );
  return response.data;
}

export async function streamResumeAgentThread(
  threadId: string,
  knowledgeBaseId: number,
  callbacks: AgentStreamCallbacks,
  signal?: AbortSignal,
): Promise<void> {
  await streamAgentEndpoint(
    `/agent/threads/${encodeURIComponent(threadId)}/resume/stream`,
    { knowledge_base_id: knowledgeBaseId },
    callbacks,
    signal,
  );
}

async function streamAgentEndpoint(
  url: string,
  payload: object,
  callbacks: AgentStreamCallbacks,
  signal?: AbortSignal,
): Promise<void> {
  const token = getAccessToken();
  const response = await fetch(url, {
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
        stage: "model",
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

    if (event === "waiting") {
      const waiting = parseWaitingResponse(parsed);
      if (waiting) callbacks.onWaiting(waiting);
      return;
    }

    if (event === "cancelled") {
      callbacks.onCancelled(
        toStringValue(parsed?.message) ?? "Agent任务已取消",
      );
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

function parseWaitingResponse(
  parsed: Record<string, unknown> | null,
): AgentWaitingResponse | null {
  const threadId = toStringValue(parsed?.thread_id);
  const rawApprovals = parsed?.approvals;
  if (!threadId || !Array.isArray(rawApprovals)) return null;

  const approvals = rawApprovals.flatMap((item) => {
    if (!item || typeof item !== "object") return [];
    const record = item as Record<string, unknown>;
    const callId = toStringValue(record.call_id);
    const toolName = toStringValue(record.tool_name);
    const reason = toStringValue(record.reason);
    if (!callId || !toolName || !reason) return [];
    return [{ call_id: callId, tool_name: toolName, reason }];
  });

  return {
    status: "waiting",
    thread_id: threadId,
    approvals,
  };
}

function parseJson(value: string): Record<string, unknown> | null {
  try {
    const result: unknown = JSON.parse(value);
    return result && typeof result === "object"
      ? (result as Record<string, unknown>)
      : null;
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
