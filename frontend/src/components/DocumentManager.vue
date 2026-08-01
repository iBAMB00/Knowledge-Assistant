<script setup lang="ts">
import {
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
  DocumentRecord,
} from "@/types/knowledge";

const props = defineProps<{
  documents: DocumentRecord[];
  loading: boolean;
  busyDocumentId?: number;
  uploadBusy: boolean;
}>();

const emit = defineEmits<{
  refresh: [];
  upload: [file: File];
  process: [documentId: number];
  embed: [documentId: number];
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
  fileInput.value?.click();
}

function onFile(event: Event): void {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];

  if (file) {
    emit("upload", file);
  }

  input.value = "";
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

function canProcess(status: string): boolean {
  return [
    "uploaded",
    "parse_failed",
    "parsed",
    "chunk_failed",
  ].includes(status.toLowerCase());
}

function canEmbed(status: string): boolean {
  return [
    "chunked",
    "embedding_failed",
  ].includes(status.toLowerCase());
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
          上传文档，并完成解析、切片和向量化。
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
          上传文档
        </button>

        <input
          ref="fileInput"
          class="visually-hidden"
          type="file"
          accept=".txt,.md,.pdf"
          @change="onFile"
        />
      </div>
    </header>

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

          <div>
            <span
              class="status-badge"
              :class="statusClass(document.status)"
            >
              {{ document.status }}
            </span>
          </div>

          <time>
            {{ formatDate(document.created_at) }}
          </time>

          <div class="document-row-actions">
            <button
              v-if="canProcess(document.status)"
              type="button"
              class="table-action"
              :disabled="
                busyDocumentId === document.id
              "
              @click="emit('process', document.id)"
            >
              <LoaderCircle
                v-if="
                  busyDocumentId === document.id
                "
                :size="16"
                class="spinning"
              />
              <WandSparkles v-else :size="16" />
              处理
            </button>

            <button
              v-if="canEmbed(document.status)"
              type="button"
              class="table-action"
              :disabled="
                busyDocumentId === document.id
              "
              @click="emit('embed', document.id)"
            >
              <LoaderCircle
                v-if="
                  busyDocumentId === document.id
                "
                :size="16"
                class="spinning"
              />
              <RefreshCw v-else :size="16" />
              向量化
            </button>

            <button
              type="button"
              class="table-action danger"
              :disabled="
                busyDocumentId === document.id
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
