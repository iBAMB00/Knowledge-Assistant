import type { AxiosProgressEvent } from "axios";
import { getAccessToken, http } from "@/api/http";
import type {
  DocumentRecord,
  KnowledgeChatRequest,
  KnowledgeChatResponse,
  KnowledgeChatSource,
  ProcessingJobRecord,
  ProcessingJobType,
  StreamCallbacks,
} from "@/types/knowledge";
import { consumeSse } from "@/utils/sse";

export async function listDocuments(knowledgeBaseId: number): Promise<DocumentRecord[]> {
  const response = await http.get<DocumentRecord[]>("/documents/", {
    params: { knowledge_base_id: knowledgeBaseId },
  });
  return response.data;
}

export async function uploadDocument(
  knowledgeBaseId: number,
  file: File,
  onProgress?: (progress: number) => void,
): Promise<DocumentRecord> {
  const form = new FormData();
  form.append("file", file);
  form.append("knowledge_base_id", String(knowledgeBaseId));

  const response = await http.post<DocumentRecord>("/documents/", form, {
    onUploadProgress: (event: AxiosProgressEvent) => {
      const total = event.total ?? file.size;
      if (total <= 0) return;
      onProgress?.(Math.min(100, Math.round((event.loaded / total) * 100)));
    },
  });

  onProgress?.(100);
  return response.data;
}

export async function createProcessingJob(
  documentId: number,
  jobType: ProcessingJobType = "full_pipeline",
): Promise<ProcessingJobRecord> {
  const response = await http.post<ProcessingJobRecord>(
    `/documents/${documentId}/processing-jobs`,
    { job_type: jobType },
  );
  return response.data;
}

export async function getProcessingJob(jobId: number): Promise<ProcessingJobRecord> {
  const response = await http.get<ProcessingJobRecord>(`/processing-jobs/${jobId}`);
  return response.data;
}

export async function getLatestProcessingJob(
  documentId: number,
): Promise<ProcessingJobRecord> {
  const response = await http.get<ProcessingJobRecord>(
    `/documents/${documentId}/processing-jobs/latest`,
  );
  return response.data;
}

export async function deleteDocument(documentId: number): Promise<void> {
  await http.delete(`/documents/${documentId}`);
}

export async function chatWithKnowledge(
  payload: KnowledgeChatRequest,
): Promise<KnowledgeChatResponse> {
  const response = await http.post<KnowledgeChatResponse>("/knowledge/chat", payload);
  return response.data;
}

export async function streamKnowledgeChat(
  payload: KnowledgeChatRequest,
  callbacks: StreamCallbacks,
  signal?: AbortSignal,
): Promise<void> {
  const token = getAccessToken();
  const response = await fetch("/knowledge/chat/stream", {
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

    if (event === "metadata") {
      callbacks.onSources(normalizeSources(parsed?.sources));
      return;
    }

    if (event === "message") {
      if (typeof parsed?.content === "string") callbacks.onContent(parsed.content);
      return;
    }

    if (event === "done") {
      callbacks.onDone();
      return;
    }

    if (event === "error") {
      throw new Error(
        typeof parsed?.message === "string" ? parsed.message : "知识库问答失败",
      );
    }
  });
}

function normalizeSources(value: unknown): KnowledgeChatSource[] {
  if (!Array.isArray(value)) return [];

  return value
    .map((item, index): KnowledgeChatSource | null => {
      if (!item || typeof item !== "object") return null;
      const source = item as Record<string, unknown>;
      const documentId = toNumber(source.document_id);
      const chunkId = toNumber(source.chunk_id);
      const excerpt = typeof source.excerpt === "string" ? source.excerpt : "";
      const filename = typeof source.filename === "string" ? source.filename : "未知文档";
      if (!documentId || !chunkId || !excerpt.trim()) return null;

      return {
        source_number: toNumber(source.source_number) ?? index + 1,
        document_id: documentId,
        filename,
        chunk_id: chunkId,
        excerpt,
        section_title:
          typeof source.section_title === "string" ? source.section_title : null,
        heading_path: Array.isArray(source.heading_path)
          ? source.heading_path.filter((x): x is string => typeof x === "string")
          : [],
        start_page: toNumber(source.start_page) ?? null,
        end_page: toNumber(source.end_page) ?? null,
        page_numbers: Array.isArray(source.page_numbers)
          ? source.page_numbers.filter((x): x is number => typeof x === "number")
          : [],
      };
    })
    .filter((item): item is KnowledgeChatSource => item !== null);
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
