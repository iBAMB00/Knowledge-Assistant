<script setup lang="ts">
import {
  CheckCircle2,
  CircleAlert,
  FilePlus2,
  FileText,
  LoaderCircle,
  RefreshCw,
  Search,
  Trash2,
  WandSparkles,
} from "lucide-vue-next";
import { computed, ref } from "vue";
import type {
  ActiveProcessingJob,
  DocumentRecord,
  KnowledgeBaseRecord,
  ProcessingJobSnapshot,
  ProcessingJobStage,
} from "@/types/knowledge";

const props = defineProps<{
  knowledgeBase: KnowledgeBaseRecord;
  documents: DocumentRecord[];
  jobsByDocumentId: Record<number, ProcessingJobSnapshot>;
  loading: boolean;
  busyDocumentId?: number;
  uploadBusy: boolean;
  uploadProgress: number | null;
}>();

const emit = defineEmits<{
  refresh: [];
  upload: [file: File];
  start: [documentId: number];
  remove: [document: DocumentRecord];
  tab: [view: "documents" | "chat" | "knowledge-base-settings"];
}>();

const keyword = ref("");
const fileInput = ref<HTMLInputElement | null>(null);

const filteredDocuments = computed(() => {
  const value = keyword.value.trim().toLowerCase();
  return value
    ? props.documents.filter((document) => document.filename.toLowerCase().includes(value))
    : props.documents;
});

function chooseFile(): void {
  if (!props.uploadBusy) fileInput.value?.click();
}

function onFile(event: Event): void {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  if (file && !props.uploadBusy) emit("upload", file);
  input.value = "";
}

function getJob(document: DocumentRecord): ProcessingJobSnapshot | ActiveProcessingJob | null {
  return props.jobsByDocumentId[document.id] ?? document.active_job ?? null;
}

function isActiveJob(job: ProcessingJobSnapshot | ActiveProcessingJob | null): boolean {
  return job?.status === "pending" || job?.status === "running";
}

function canStart(status: string): boolean {
  return ["uploaded", "parse_failed", "parsed", "chunk_failed", "chunked", "embedding_failed"].includes(status.toLowerCase());
}

function isRetry(document: DocumentRecord): boolean {
  const job = getJob(document);
  return job?.status === "failed" || document.status.toLowerCase().endsWith("_failed");
}

function stageLabel(stage: ProcessingJobStage): string {
  const labels: Record<ProcessingJobStage, string> = {
    queued: "等待处理",
    parsing: "正在解析文档",
    chunking: "正在生成切片",
    embedding: "正在生成向量",
    indexing: "正在同步检索索引",
    finalizing: "正在完成处理",
    completed: "处理完成",
  };
  return labels[stage];
}

function documentStatusLabel(status: string): string {
  const labels: Record<string, string> = {
    uploaded: "已上传",
    parsing: "解析中",
    parsed: "已解析",
    parse_failed: "解析失败",
    chunking: "切片中",
    chunked: "已切片",
    chunk_failed: "切片失败",
    embedding: "向量化中",
    embedding_failed: "向量化失败",
    completed: "已完成",
  };
  return labels[status.toLowerCase()] ?? status;
}

function formatSize(size: number): string {
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
  return `${(size / 1024 / 1024).toFixed(1)} MB`;
}

function formatDate(value?: string): string {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat("zh-CN", {
    year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit",
  }).format(date);
}

function statusClass(status: string): string {
  return `status-${status.toLowerCase().replaceAll("_", "-")}`;
}
</script>

<template>
  <section class="page-shell">
    <div class="breadcrumb">知识库 <span>›</span> {{ knowledgeBase.name }}</div>
    <header class="kb-detail-heading">
      <div>
        <h1>{{ knowledgeBase.name }}</h1>
        <p>{{ knowledgeBase.description || '暂无描述' }}</p>
        <div class="kb-meta"><span>文档数量：{{ documents.length }}</span><span>知识库 ID：{{ knowledgeBase.id }}</span><span>Owner：{{ knowledgeBase.owner_id }}</span></div>
      </div>
      <button type="button" class="primary-button" :disabled="uploadBusy" @click="chooseFile">
        <LoaderCircle v-if="uploadBusy" :size="17" class="spinning" />
        <FilePlus2 v-else :size="17" />
        {{ uploadBusy ? `上传中 ${uploadProgress ?? 0}%` : '上传文档' }}
      </button>
      <input ref="fileInput" class="visually-hidden" type="file" accept=".txt,.md,.markdown,.pdf,.html,.htm" :disabled="uploadBusy" @change="onFile" />
    </header>

    <nav class="detail-tabs">
      <button class="active" type="button">文档管理</button>
      <button type="button" @click="emit('tab', 'chat')">聊天问答</button>
      <button type="button" @click="emit('tab', 'knowledge-base-settings')">知识库设置</button>
    </nav>

    <div v-if="uploadBusy" class="upload-progress-card">
      <div class="progress-heading"><span>正在上传原始文件</span><strong>{{ uploadProgress ?? 0 }}%</strong></div>
      <div class="progress-track"><i :style="{ width: `${uploadProgress ?? 0}%` }" /></div>
      <small>上传成功后可启动完整处理任务：解析 → 切片 → 向量化 → 索引。</small>
    </div>

    <div class="toolbar-row">
      <button type="button" class="primary-button compact" :disabled="uploadBusy" @click="chooseFile"><FilePlus2 :size="16" />上传文档</button>
      <label class="search-box"><Search :size="16" /><input v-model="keyword" type="search" placeholder="搜索文档..." /></label>
      <button type="button" class="secondary-button compact" :disabled="loading" @click="emit('refresh')"><RefreshCw :size="16" :class="{ spinning: loading }" />刷新</button>
      <span class="toolbar-spacer" />
      <span class="muted small-text">共 {{ filteredDocuments.length }} 份文档</span>
    </div>

    <div class="table-card document-table">
      <div class="table-head document-grid"><span>文件名</span><span>大小</span><span>状态</span><span>处理进度</span><span>上传时间</span><span>操作</span></div>

      <div v-if="loading && documents.length === 0" class="empty-state"><LoaderCircle :size="28" class="spinning" /><p>正在加载文档…</p></div>
      <div v-else-if="filteredDocuments.length === 0" class="empty-state"><FileText :size="38" /><p>暂无文档，请上传第一份知识文件。</p></div>

      <article v-for="document in filteredDocuments" :key="document.id" class="table-row document-grid">
        <div class="document-name-cell"><span class="table-leading-icon"><FileText :size="18" /></span><div><strong>{{ document.filename }}</strong><small>ID {{ document.id }}</small></div></div>
        <span class="muted">{{ formatSize(document.size) }}</span>
        <span><i class="status-pill" :class="statusClass(document.status)">{{ documentStatusLabel(document.status) }}</i></span>
        <div class="processing-cell">
          <template v-if="getJob(document)">
            <div class="progress-heading compact-heading">
              <span>
                <LoaderCircle v-if="isActiveJob(getJob(document))" :size="13" class="spinning" />
                <CircleAlert v-else-if="getJob(document)?.status === 'failed'" :size="13" />
                <CheckCircle2 v-else :size="13" />
                {{ stageLabel(getJob(document)?.stage ?? 'queued') }}
              </span>
              <strong>{{ getJob(document)?.progress ?? 0 }}%</strong>
            </div>
            <div class="progress-track mini"><i :class="`job-${getJob(document)?.status}`" :style="{ width: `${getJob(document)?.progress ?? 0}%` }" /></div>
            <small v-if="getJob(document)?.status === 'failed'" class="danger-text">{{ getJob(document)?.error_message || '处理失败，请重试。' }}</small>
          </template>
          <span v-else class="muted small-text">暂无任务</span>
        </div>
        <span class="muted small-text">{{ formatDate(document.created_at) }}</span>
        <div class="row-actions">
          <button v-if="canStart(document.status)" type="button" class="icon-button" :disabled="busyDocumentId === document.id || isActiveJob(getJob(document))" :title="isRetry(document) ? '重试处理' : '开始处理'" @click="emit('start', document.id)">
            <LoaderCircle v-if="busyDocumentId === document.id" :size="15" class="spinning" />
            <RefreshCw v-else-if="isRetry(document)" :size="15" />
            <WandSparkles v-else :size="15" />
          </button>
          <button type="button" class="icon-button danger" :disabled="busyDocumentId === document.id || isActiveJob(getJob(document))" title="删除文档" @click="emit('remove', document)"><Trash2 :size="15" /></button>
        </div>
      </article>
    </div>
  </section>
</template>
