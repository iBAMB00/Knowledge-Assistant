import { http } from "@/api/http";
import type { KnowledgeBaseRecord } from "@/types/knowledge";

export async function listKnowledgeBases(): Promise<KnowledgeBaseRecord[]> {
  const response = await http.get<KnowledgeBaseRecord[]>("/knowledge-bases/");
  return response.data;
}

export async function getKnowledgeBase(id: number): Promise<KnowledgeBaseRecord> {
  const response = await http.get<KnowledgeBaseRecord>(`/knowledge-bases/${id}`);
  return response.data;
}

export async function createKnowledgeBase(
  name: string,
  description: string | null,
): Promise<KnowledgeBaseRecord> {
  const response = await http.post<KnowledgeBaseRecord>("/knowledge-bases/", {
    name,
    description: description || null,
  });
  return response.data;
}

export async function updateKnowledgeBase(
  id: number,
  payload: { name?: string; description?: string | null },
): Promise<KnowledgeBaseRecord> {
  const response = await http.patch<KnowledgeBaseRecord>(
    `/knowledge-bases/${id}`,
    payload,
  );
  return response.data;
}

export async function deleteKnowledgeBase(id: number): Promise<void> {
  await http.delete(`/knowledge-bases/${id}`);
}
