<script setup lang="ts">
import {
  Bot,
  CheckCircle2,
  ChevronDown,
  Moon,
  RotateCcw,
  Sun,
  Trash2,
} from "lucide-vue-next";
import {
  computed,
  nextTick,
  onMounted,
  ref,
} from "vue";
import AppSidebar from "@/components/AppSidebar.vue";
import ChatComposer from "@/components/ChatComposer.vue";
import ChatMessage from "@/components/ChatMessage.vue";
import DocumentManager from "@/components/DocumentManager.vue";
import {
  createProcessingJob,
  deleteDocument,
  getChunkSummary,
  listDocuments,
  uploadDocument,
} from "@/api/knowledge";
import { getApiErrorMessage } from "@/api/http";
import { useKnowledgeChat } from "@/composables/useKnowledgeChat";
import {
  isActiveProcessingJob,
  useProcessingJobPolling,
} from "@/composables/useProcessingJobPolling";
import type {
  DocumentRecord,
  KnowledgeStats,
  ProcessingJobSnapshot,
} from "@/types/knowledge";

type ViewKey = "chat" | "documents";

interface RefreshDocumentOptions {
  showLoading?: boolean;
  refreshChunks?: boolean;
}

const activeView = ref<ViewKey>("chat");
const darkMode = ref(false);
const documents = ref<DocumentRecord[]>([]);
const selectedDocumentId = ref<number>();
const documentLoading = ref(false);
const uploadBusy = ref(false);
const uploadProgress = ref<number | null>(null);
const busyDocumentId = ref<number>();
const chatViewport =
  ref<HTMLElement | null>(null);
const chunkTotals =
  ref<Record<number, number>>({});
const notice = ref<{
  type: "success" | "error";
  message: string;
} | null>(null);

const {
  jobsByDocumentId,
  syncDocuments,
  trackJob,
  forgetDocument,
} = useProcessingJobPolling({
  onTerminalJobs: handleTerminalJobs,
  onPollError: () => {
    showNotice(
      "error",
      "任务状态刷新暂时失败，系统将降低频率后自动重试。",
    );
  },
});

const {
  messages,
  submitting,
  streamingEnabled,
  sendQuestion,
  stopGeneration,
  clearConversation,
} = useKnowledgeChat(scrollToBottom);

const stats = computed<KnowledgeStats>(() => {
  const chunkCount = Object.values(
    chunkTotals.value,
  ).reduce((total, value) => total + value, 0);

  const completedIds = new Set(
    documents.value
      .filter((document) =>
        ["completed", "embedded"].includes(
          document.status.toLowerCase(),
        ),
      )
      .map((document) => document.id),
  );

  const vectorChunkCount = Object.entries(
    chunkTotals.value,
  ).reduce(
    (total, [documentId, count]) =>
      completedIds.has(Number(documentId))
        ? total + count
        : total,
    0,
  );

  const newest = documents.value
    .map((document) => document.created_at)
    .filter(
      (value): value is string =>
        typeof value === "string",
    )
    .sort()
    .at(-1);

  return {
    documentCount: documents.value.length,
    chunkCount,
    vectorChunkCount,
    lastUpdated: newest,
  };
});

const selectedDocumentName = computed(() => {
  if (!selectedDocumentId.value) {
    return "全部知识库";
  }

  return (
    documents.value.find(
      (document) =>
        document.id === selectedDocumentId.value,
    )?.filename ?? "指定文档"
  );
});

onMounted(async () => {
  darkMode.value =
    localStorage.getItem(
      "knowledge-assistant-theme",
    ) === "dark";
  applyTheme();
  await refreshDocuments();
});

async function refreshDocuments(
  options: RefreshDocumentOptions = {},
): Promise<void> {
  const showLoading = options.showLoading ?? true;
  const refreshChunks = options.refreshChunks ?? true;

  if (showLoading) {
    documentLoading.value = true;
  }

  try {
    documents.value = await listDocuments();
    syncDocuments(documents.value);

    if (
      selectedDocumentId.value &&
      !documents.value.some(
        (document) =>
          document.id === selectedDocumentId.value,
      )
    ) {
      selectedDocumentId.value = undefined;
    }

    if (refreshChunks) {
      await refreshChunkTotals();
    }
  } catch (error) {
    showNotice("error", getApiErrorMessage(error));
  } finally {
    if (showLoading) {
      documentLoading.value = false;
    }
  }
}

async function refreshChunkTotals(
  documentIds?: number[],
): Promise<void> {
  const targetIds = documentIds
    ? new Set(documentIds)
    : null;

  const targets = targetIds
    ? documents.value.filter((document) =>
        targetIds.has(document.id),
      )
    : documents.value;

  const settled = await Promise.allSettled(
    targets.map(async (document) => ({
      documentId: document.id,
      summary: await getChunkSummary(document.id),
    })),
  );

  const totals = targetIds
    ? { ...chunkTotals.value }
    : {};

  for (const result of settled) {
    if (result.status === "fulfilled") {
      totals[result.value.documentId] =
        result.value.summary.total_chunks;
    }
  }

  chunkTotals.value = totals;
}

async function handleUpload(
  file: File,
): Promise<void> {
  if (uploadBusy.value) {
    return;
  }

  uploadBusy.value = true;
  uploadProgress.value = 0;
  let uploadedDocument: DocumentRecord | undefined;

  try {
    uploadedDocument = await uploadDocument(
      file,
      (progress) => {
        uploadProgress.value = progress;
      },
    );

    showNotice(
      "success",
      `“${file.name}”上传完成，正在创建后台处理任务。`,
    );
  } catch (error) {
    showNotice("error", getApiErrorMessage(error));
  } finally {
    uploadBusy.value = false;
    uploadProgress.value = null;
  }

  if (!uploadedDocument) {
    return;
  }

  await refreshDocuments({
    showLoading: false,
    refreshChunks: false,
  });

  await handleStartProcessing(
    uploadedDocument.id,
    `“${file.name}”已进入后台处理队列。`,
  );
}

async function handleStartProcessing(
  documentId: number,
  successMessage = "后台处理任务已创建。",
): Promise<void> {
  const currentJob =
    jobsByDocumentId.value[documentId] ??
    documents.value.find(
      (document) => document.id === documentId,
    )?.active_job;

  if (
    busyDocumentId.value === documentId ||
    (currentJob && isActiveProcessingJob(currentJob))
  ) {
    return;
  }

  busyDocumentId.value = documentId;

  try {
    const job = await createProcessingJob(
      documentId,
      "full_pipeline",
    );

    trackJob(job);
    showNotice("success", successMessage);

    await refreshDocuments({
      showLoading: false,
      refreshChunks: false,
    });
  } catch (error) {
    showNotice("error", getApiErrorMessage(error));

    await refreshDocuments({
      showLoading: false,
      refreshChunks: false,
    });
  } finally {
    busyDocumentId.value = undefined;
  }
}

async function handleTerminalJobs(
  jobs: ProcessingJobSnapshot[],
): Promise<void> {
  await refreshDocuments({
    showLoading: false,
    refreshChunks: false,
  });

  await refreshChunkTotals(
    jobs.map((job) => job.document_id),
  );

  const failedJob = jobs.find(
    (job) => job.status === "failed",
  );

  if (failedJob) {
    showNotice(
      "error",
      failedJob.error_message ??
        "文档处理失败，请检查任务状态后重试。",
    );
    return;
  }

  if (jobs.length === 1) {
    const document = documents.value.find(
      (item) => item.id === jobs[0].document_id,
    );

    showNotice(
      "success",
      document
        ? `“${document.filename}”处理完成。`
        : "文档处理完成。",
    );
    return;
  }

  showNotice(
    "success",
    `${jobs.length} 份文档处理完成。`,
  );
}

async function handleDelete(
  documentRecord: DocumentRecord,
): Promise<void> {
  const currentJob =
    jobsByDocumentId.value[documentRecord.id] ??
    documentRecord.active_job;

  if (currentJob && isActiveProcessingJob(currentJob)) {
    showNotice(
      "error",
      "文档仍有活动处理任务，暂时不能删除。",
    );
    return;
  }

  if (
    !window.confirm(
      `确认删除“${documentRecord.filename}”吗？`,
    )
  ) {
    return;
  }

  busyDocumentId.value = documentRecord.id;

  try {
    await deleteDocument(documentRecord.id);
    forgetDocument(documentRecord.id);
    delete chunkTotals.value[documentRecord.id];

    await refreshDocuments({
      showLoading: false,
      refreshChunks: false,
    });

    showNotice("success", "文档已删除。");
  } catch (error) {
    showNotice("error", getApiErrorMessage(error));
  } finally {
    busyDocumentId.value = undefined;
  }
}

function navigate(view: ViewKey): void {
  activeView.value = view;
}

function resetChat(): void {
  if (
    messages.value.length > 1 &&
    !window.confirm("确认清空当前对话吗？")
  ) {
    return;
  }

  clearConversation();
}

function toggleTheme(): void {
  darkMode.value = !darkMode.value;
  localStorage.setItem(
    "knowledge-assistant-theme",
    darkMode.value ? "dark" : "light",
  );
  applyTheme();
}

function applyTheme(): void {
  document.documentElement.dataset.theme =
    darkMode.value ? "dark" : "light";
}

function showNotice(
  type: "success" | "error",
  message: string,
): void {
  notice.value = { type, message };

  window.setTimeout(() => {
    if (notice.value?.message === message) {
      notice.value = null;
    }
  }, 4000);
}

async function scrollToBottom(): Promise<void> {
  await nextTick();

  if (chatViewport.value) {
    chatViewport.value.scrollTop =
      chatViewport.value.scrollHeight;
  }
}
</script>

<template>
  <div class="app-shell">
    <AppSidebar
      :active-view="activeView"
      :stats="stats"
      @navigate="navigate"
    />

    <main class="workspace">
      <header class="topbar">
        <div>
          <p class="eyebrow">
            {{
              activeView === "chat"
                ? "Enterprise RAG"
                : "Knowledge Operations"
            }}
          </p>
          <h2>
            {{
              activeView === "chat"
                ? "智能对话"
                : "知识库管理"
            }}
          </h2>
        </div>

        <div class="topbar-actions">
          <button
            type="button"
            class="icon-button top-icon-button"
            :title="
              darkMode
                ? '切换浅色模式'
                : '切换深色模式'
            "
            @click="toggleTheme"
          >
            <Sun v-if="darkMode" :size="19" />
            <Moon v-else :size="19" />
          </button>

          <div
            v-if="activeView === 'chat'"
            class="model-picker"
          >
            <Bot :size="17" />
            <span>企业知识问答</span>
            <ChevronDown :size="15" />
          </div>

          <button
            v-if="activeView === 'chat'"
            type="button"
            class="secondary-button"
            @click="resetChat"
          >
            <Trash2 :size="17" />
            清空对话
          </button>
        </div>
      </header>

      <section
        v-if="activeView === 'chat'"
        class="chat-page"
      >
        <div class="chat-context-bar">
          <span>当前检索范围</span>
          <strong>{{ selectedDocumentName }}</strong>
          <i />
          <span>
            {{
              streamingEnabled
                ? "SSE 流式回答"
                : "普通回答"
            }}
          </span>
        </div>

        <div
          ref="chatViewport"
          class="chat-viewport"
        >
          <div class="messages">
            <ChatMessage
              v-for="message in messages"
              :key="message.id"
              :message="message"
              :documents="documents"
            />
          </div>
        </div>

        <ChatComposer
          v-model:selected-document-id="
            selectedDocumentId
          "
          v-model:streaming-enabled="
            streamingEnabled
          "
          :documents="documents"
          :submitting="submitting"
          :upload-busy="uploadBusy"
          :upload-progress="uploadProgress"
          @send="
            (question) =>
              sendQuestion(
                question,
                selectedDocumentId,
              )
          "
          @stop="stopGeneration"
          @upload="handleUpload"
        />
      </section>

      <DocumentManager
        v-else
        :documents="documents"
        :jobs-by-document-id="jobsByDocumentId"
        :loading="documentLoading"
        :busy-document-id="busyDocumentId"
        :upload-busy="uploadBusy"
        :upload-progress="uploadProgress"
        @refresh="refreshDocuments"
        @upload="handleUpload"
        @start="handleStartProcessing"
        @remove="handleDelete"
      />

      <footer class="app-footer">
        Knowledge Assistant v0.1.0
        <span />
        基于 RAG 和大语言模型构建
      </footer>
    </main>

    <Transition name="notice">
      <div
        v-if="notice"
        class="toast"
        :class="`toast-${notice.type}`"
      >
        <CheckCircle2
          v-if="notice.type === 'success'"
          :size="19"
        />
        <RotateCcw v-else :size="19" />
        {{ notice.message }}
      </div>
    </Transition>
  </div>
</template>
