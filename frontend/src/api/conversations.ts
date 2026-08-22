import { http } from "@/api/http";
import type {
  ConversationBackendMode,
  ConversationCreatePayload,
  ConversationMessageRecord,
  ConversationRecord,
} from "@/types/knowledge";

export interface ConversationListParams {
  mode?: ConversationBackendMode;
  knowledgeBaseId?: number;
  limit?: number;
}

export async function createConversation(
  payload: ConversationCreatePayload,
): Promise<ConversationRecord> {
  const response = await http.post<ConversationRecord>("/conversations/", payload);
  return response.data;
}

export async function listConversations(
  params: ConversationListParams = {},
): Promise<ConversationRecord[]> {
  const response = await http.get<ConversationRecord[]>("/conversations/", {
    params: {
      ...(params.mode ? { mode: params.mode } : {}),
      ...(params.knowledgeBaseId
        ? { knowledge_base_id: params.knowledgeBaseId }
        : {}),
      limit: params.limit ?? 50,
    },
  });
  return response.data;
}

export async function listConversationMessages(
  conversationId: number,
  limit = 500,
): Promise<ConversationMessageRecord[]> {
  const response = await http.get<ConversationMessageRecord[]>(
    `/conversations/${conversationId}/messages`,
    { params: { limit } },
  );
  return response.data;
}

export async function deleteConversation(conversationId: number): Promise<void> {
  await http.delete(`/conversations/${conversationId}`);
}
