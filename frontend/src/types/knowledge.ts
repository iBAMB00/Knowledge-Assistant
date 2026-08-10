export interface UserRecord {
  id: number;
  email: string;
  role: string;
  is_active: boolean;
  created_at: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
}

export interface KnowledgeBaseRecord {
  id: number;
  owner_id: number;
  name: string;
  description: string | null;
  created_at: string;
  updated_at: string;
}

export interface KnowledgeChatSource {
  source_number: number;
  document_id: number;
  filename: string;
  chunk_id: number;
  excerpt: string;
  section_title: string | null;
  heading_path: string[];
  start_page: number | null;
  end_page: number | null;
  page_numbers: number[];
}

export interface KnowledgeChatResponse {
  answer: string;
  sources: KnowledgeChatSource[];
}

export interface KnowledgeChatRequest {
  question: string;
  knowledge_base_id: number;
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

export type DocumentStatus =
  | "uploaded"
  | "parsing"
  | "parsed"
  | "parse_failed"
  | "chunking"
  | "chunked"
  | "chunk_failed"
  | "embedding"
  | "embedding_failed"
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

export interface ProcessingJobRecord extends ActiveProcessingJob {
  document_id: number;
  created_at: string;
  updated_at: string;
  finished_at: string | null;
}

export interface ProcessingJobSnapshot extends ActiveProcessingJob {
  document_id: number;
  created_at?: string;
  updated_at?: string;
  finished_at?: string | null;
}

export interface DocumentRecord {
  id: number;
  knowledge_base_id: number;
  filename: string;
  size: number;
  status: DocumentStatus;
  created_at: string;
  active_job?: ActiveProcessingJob | null;
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

export interface StreamCallbacks {
  onSources: (sources: KnowledgeChatSource[]) => void;
  onContent: (content: string) => void;
  onDone: () => void;
}

export type AppView =
  | "knowledge-bases"
  | "documents"
  | "processing"
  | "chat"
  | "knowledge-base-settings"
  | "profile";
