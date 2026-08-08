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
  ProcessingJobSnapshot,
  ProcessingJobStage,
} from "@/types/knowledge";

const props = defineProps<{
  documents: DocumentRecord[];
  jobsByDocumentId: Record<
    number,
    ProcessingJobSnapshot
  >;
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
}>();

const keyword = ref("");
const fileInput =
  ref<HTMLInputElement | null>(null);

const filteredDocuments = computed(() => {
  const value = keyword.value
    .trim()
    .toLowerCase();

  return value
    ? props.documents.filter((document) =>
        document.filename
          .toLowerCase()
          .includes(value),
      )
    : props.documents;
});

function chooseFile(): void {
  if (!props.uploadBusy) {
    fileInput.value?.click();
  }
}

function onFile(event: Event): void {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];

  if (file && !props.uploadBusy) {
    emit("upload", file);
  }

  input.value = "";
}

function getJob(
  document: DocumentRecord,
): ProcessingJobSnapshot | ActiveProcessingJob | null {
  return (
    props.jobsByDocumentId[document.id] ??
    document.active_job ??
    null
  );
}

function isActiveJob(
  job: ProcessingJobSnapshot | ActiveProcessingJob | null,
): boolean {
  return (
    job?.status === "pending" ||
    job?.status === "running"
  );
}

function canStart(status: string): boolean {
  return [
    "uploaded",
    "parse_failed",
    "parsed",
    "chunk_failed",
    "chunked",
    "embedding_failed",
  ].includes(status.toLowerCase());
}

function isRetry(
  document: DocumentRecord,
): boolean {
  const job = getJob(document);

  return (
    job?.status === "failed" ||
    document.status.toLowerCase().endsWith("_failed")
  );
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

function jobStateClass(
  job: ProcessingJobSnapshot | ActiveProcessingJob | null,
): string {
  return job ? `processing-job-${job.status}` : "";
}

function formatSize(size: number): string {
  if (size < 1024) {
    return `${size} B`;
  }

  if (size < 1024 * 1024) {
    return `${(size / 1024).toFixed(1)} KB`;
  }

  return `${(size / 1024 / 1024).toFixed(1)} MB`;
}

function formatDate(value?: string): string {
  if (!value) {
    return "—";
  }

  const date = new Date(value);

  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return new Intl.DateTimeFormat("zh-CN", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
}

function statusClass(status: string): string {
  return `status-${status
    .toLowerCase()
    .replaceAll("_", "-")}`;
}
</script>

<template>
  <section class="document-page">
    <header class="document-header">
      <div>
        <p class="eyebrow">Knowledge Base</p>
        <h2>知识库管理</h2>
        <span>
          上传文档，并在后台完成解析、切片、向量化和索引。
        </span>
      </div>

      <div class="document-header-actions">
        <button
          type="button"
          class="secondary-button"
          :disabled="loading"
          @click="emit('refresh')"
        >
          <RefreshCw
            :size="17"
            :class="{ spinning: loading }"
          />
          刷新
        </button>

        <button
          type="button"
          class="primary-button"
          :disabled="uploadBusy"
          @click="chooseFile"
        >
          <LoaderCircle
            v-if="uploadBusy"
            :size="17"
            class="spinning"
          />
          <FilePlus2 v-else :size="17" />
          {{
            uploadBusy
              ? `上传中 ${uploadProgress ?? 0}%`
              : "上传文档"
          }}
        </button>

        <input
          ref="fileInput"
          class="visually-hidden"
          type="file"
          accept=".txt,.md,.pdf"
          :disabled="uploadBusy"
          @change="onFile"
        />
      </div>
    </header>

    <div
      v-if="uploadBusy"
      class="upload-progress-card"
    >
      <div class="upload-progress-heading">
        <span>正在上传原始文件</span>
        <strong>{{ uploadProgress ?? 0 }}%</strong>
      </div>
      <div class="upload-progress-track">
        <i
          :style="{
            width: `${uploadProgress ?? 0}%`,
          }"
        />
      </div>
      <small>
        上传完成后会创建后台处理任务；处理进度将在对应文档行中单独显示。
      </small>
    </div>

    <div class="document-toolbar">
      <label class="search-box">
        <Search :size="17" />
        <input
          v-model="keyword"
          type="search"
          placeholder="搜索文档名称"
        />
      </label>

      <span>
        共 {{ filteredDocuments.length }} 份文档
      </span>
    </div>

    <div class="document-table-card">
      <div
        v-if="loading && documents.length === 0"
        class="empty-state"
      >
        <LoaderCircle
          :size="28"
          class="spinning"
        />
        <p>正在加载知识库…</p>
      </div>

      <div
        v-else-if="filteredDocuments.length === 0"
        class="empty-state"
      >
        <FileText :size="34" />
        <p>暂无文档，请先上传知识文件。</p>
      </div>

      <div v-else class="document-list">
        <article
          v-for="document in filteredDocuments"
          :key="document.id"
          class="document-row"
        >
          <div class="document-file-icon">
            <FileText :size="22" />
          </div>

          <div class="document-name">
            <strong>{{ document.filename }}</strong>
            <span>
              ID {{ document.id }} ·
              {{ formatSize(document.size) }}
            </span>
          </div>

          <div class="document-status-cell">
            <span
              class="status-badge"
              :class="statusClass(document.status)"
            >
              {{ documentStatusLabel(document.status) }}
            </span>
          </div>

          <div class="document-processing">
            <template v-if="getJob(document)">
              <div
                class="processing-job-heading"
                :class="jobStateClass(getJob(document))"
              >
                <LoaderCircle
                  v-if="isActiveJob(getJob(document))"
                  :size="15"
                  class="spinning"
                />
                <CircleAlert
                  v-else-if="getJob(document)?.status === 'failed'"
                  :size="15"
                />
                <CheckCircle2 v-else :size="15" />

                <span>
                  {{ stageLabel(getJob(document)?.stage ?? "queued") }}
                </span>
                <strong>
                  {{ getJob(document)?.progress ?? 0 }}%
                </strong>
              </div>

              <div class="processing-progress-track">
                <i
                  :class="jobStateClass(getJob(document))"
                  :style="{
                    width: `${getJob(document)?.progress ?? 0}%`,
                  }"
                />
              </div>

              <small
                v-if="
                  getJob(document)?.status === 'failed'
                "
                class="processing-job-message"
              >
                {{
                  getJob(document)?.error_message ??
                  "处理失败，请重试。"
                }}
              </small>
            </template>

            <span v-else class="processing-job-idle">
              暂无后台任务
            </span>
          </div>

          <time>
            {{ formatDate(document.created_at) }}
          </time>

          <div class="document-row-actions">
            <button
              v-if="canStart(document.status)"
              type="button"
              class="table-action"
              :disabled="
                busyDocumentId === document.id ||
                isActiveJob(getJob(document))
              "
              @click="emit('start', document.id)"
            >
              <LoaderCircle
                v-if="busyDocumentId === document.id"
                :size="16"
                class="spinning"
              />
              <RefreshCw
                v-else-if="isRetry(document)"
                :size="16"
              />
              <WandSparkles v-else :size="16" />
              {{ isRetry(document) ? "重试" : "开始处理" }}
            </button>

            <button
              type="button"
              class="table-action danger"
              :disabled="
                busyDocumentId === document.id ||
                isActiveJob(getJob(document))
              "
              :title="
                isActiveJob(getJob(document))
                  ? '活动任务期间不能删除文档'
                  : '删除文档'
              "
              @click="emit('remove', document)"
            >
              <Trash2 :size="16" />
            </button>
          </div>
        </article>
      </div>
    </div>
  </section>
</template>
