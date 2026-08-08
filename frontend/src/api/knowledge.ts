import type { AxiosProgressEvent } from "axios";
import { http } from "@/api/http";
import type {
  ChunkSummary,
  DocumentRecord,
  KnowledgeChatRequest,
  KnowledgeChatResponse,
  KnowledgeChatSource,
  ProcessingJobRecord,
  ProcessingJobType,
  StreamCallbacks,
} from "@/types/knowledge";
import { consumeSse } from "@/utils/sse";

export async function listDocuments(): Promise<DocumentRecord[]> {
  const response = await http.get<DocumentRecord[]>(
    "/documents/",
  );
  return response.data;
}

export async function uploadDocument(
  file: File,
  onProgress?: (progress: number) => void,
): Promise<DocumentRecord> {
  const form = new FormData();
  form.append("file", file);

  const response = await http.post<DocumentRecord>(
    "/documents/",
    form,
    {
      onUploadProgress: (event: AxiosProgressEvent) => {
        const total = event.total ?? file.size;

        if (total <= 0) {
          return;
        }

        onProgress?.(
          Math.min(
            100,
            Math.round((event.loaded / total) * 100),
          ),
        );
      },
    },
  );

  onProgress?.(100);
  return response.data;
}

export async function createProcessingJob(
  documentId: number,
  jobType: ProcessingJobType = "full_pipeline",
): Promise<ProcessingJobRecord> {
  const response = await http.post<ProcessingJobRecord>(
    `/documents/${documentId}/processing-jobs`,
    {
      job_type: jobType,
    },
  );

  return response.data;
}

export async function getProcessingJob(
  jobId: number,
): Promise<ProcessingJobRecord> {
  const response = await http.get<ProcessingJobRecord>(
    `/processing-jobs/${jobId}`,
  );

  return response.data;
}

export async function deleteDocument(
  documentId: number,
): Promise<void> {
  await http.delete(`/documents/${documentId}`);
}

export async function getChunkSummary(
  documentId: number,
): Promise<ChunkSummary> {
  const response = await http.get<Record<string, unknown>>(
    `/documents/${documentId}/chunk-summary`,
  );
  const data = response.data;

  return {
    total_chunks:
      toNumber(data.total_chunks) ??
      toNumber(data.chunk_count) ??
      toNumber(data.total) ??
      0,
    total_tokens:
      toNumber(data.total_tokens) ??
      toNumber(data.token_count),
  };
}

export async function chatWithKnowledge(
  payload: KnowledgeChatRequest,
): Promise<KnowledgeChatResponse> {
  const response = await http.post<KnowledgeChatResponse>(
    "/knowledge/chat",
    payload,
  );
  return response.data;
}

export async function streamKnowledgeChat(
  payload: KnowledgeChatRequest,
  callbacks: StreamCallbacks,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(
    "/knowledge/chat/stream",
    {
      method: "POST",
      headers: {
        Accept: "text/event-stream",
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
      signal,
    },
  );

  if (!response.ok) {
    throw new Error(await readResponseError(response));
  }

  await consumeSse(response, ({ event, data }) => {
    const parsed = parseJson(data);

    if (event === "metadata") {
      callbacks.onSources(
        normalizeSources(parsed?.sources),
      );
      return;
    }

    if (event === "message") {
      if (typeof parsed?.content === "string") {
        callbacks.onContent(parsed.content);
      }
      return;
    }

    if (event === "done") {
      callbacks.onDone();
      return;
    }

    if (event === "error") {
      throw new Error(
        typeof parsed?.message === "string"
          ? parsed.message
          : "知识库问答失败",
      );
    }
  });
}

function normalizeSources(
  value: unknown,
): KnowledgeChatSource[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value
    .map((item, index): KnowledgeChatSource | null => {
      if (!item || typeof item !== "object") {
        return null;
      }

      const source = item as Record<string, unknown>;
      const documentId = toNumber(source.document_id);
      const sourceNumber =
        toNumber(source.source_number) ?? index + 1;
      const excerpt =
        typeof source.excerpt === "string"
          ? source.excerpt
          : typeof source.content === "string"
            ? source.content
            : "";

      if (!documentId || !excerpt.trim()) {
        return null;
      }

      return {
        source_number: sourceNumber,
        document_id: documentId,
        excerpt,
      };
    })
    .filter(
      (item): item is KnowledgeChatSource =>
        item !== null,
    );
}

function parseJson(
  value: string,
): Record<string, unknown> | null {
  try {
    const result: unknown = JSON.parse(value);
    return result && typeof result === "object"
      ? (result as Record<string, unknown>)
      : null;
  } catch {
    return null;
  }
}

async function readResponseError(
  response: Response,
): Promise<string> {
  const text = await response.text();

  try {
    const parsed = JSON.parse(text) as {
      detail?: unknown;
    };

    if (typeof parsed.detail === "string") {
      return parsed.detail;
    }
  } catch {
    // 使用原始错误文本。
  }

  return text || `请求失败（HTTP ${response.status}）`;
}

function toNumber(value: unknown): number | undefined {
  return typeof value === "number" &&
    Number.isFinite(value)
    ? value
    : undefined;
}
