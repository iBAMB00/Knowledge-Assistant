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

export interface DocumentRecord {
  id: number;
  filename: string;
  size: number;
  status: string;
  created_at?: string;
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
