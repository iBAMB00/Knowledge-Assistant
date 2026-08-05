export interface KnowledgeChatSource {
  source_number: number;
  document_id: number;
  excerpt: string;
}

export interface KnowledgeChatResponse {
  answer: string;
  sources: KnowledgeChatSource[];
}

export interface KnowledgeChatRequest {
  question: string;
  top_k?: number;
  document_id?: number;
}

export type ProcessingJobType =
  | "document_processing"
  | "embedding"
  | "full_pipeline";

export type ProcessingJobStatus =
  | "pending"
  | "running"
  | "succeeded"
  | "failed";

export type ProcessingJobStage =
  | "queued"
  | "parsing"
  | "chunking"
  | "embedding"
  | "indexing"
  | "finalizing"
  | "completed";

export interface ActiveProcessingJob {
  id: number;
  job_type: ProcessingJobType;
  status: ProcessingJobStatus;
  stage: ProcessingJobStage;
  progress: number;
  error_message: string | null;
  started_at: string | null;
}

export interface ProcessingJobRecord
  extends ActiveProcessingJob {
  document_id: number;
  created_at: string;
  updated_at: string;
  finished_at: string | null;
}

export interface ProcessingJobSnapshot
  extends ActiveProcessingJob {
  document_id: number;
  created_at?: string;
  updated_at?: string;
  finished_at?: string | null;
}

export interface DocumentRecord {
  id: number;
  filename: string;
  stored_name?: string;
  size: number;
  status: string;
  created_at?: string;
  active_job?: ActiveProcessingJob | null;
}

export interface ChunkSummary {
  total_chunks: number;
  total_tokens?: number;
}

export interface ChatMessageRecord {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources: KnowledgeChatSource[];
  createdAt: Date;
  pending?: boolean;
  error?: boolean;
  elapsedMs?: number;
}

export interface KnowledgeStats {
  documentCount: number;
  chunkCount: number;
  vectorChunkCount: number;
  lastUpdated?: string;
}

export interface StreamCallbacks {
  onSources: (sources: KnowledgeChatSource[]) => void;
  onContent: (content: string) => void;
  onDone: () => void;
}
