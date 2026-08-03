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
  createDocumentEmbeddings,
  deleteDocument,
  getChunkSummary,
  listDocuments,
  processDocument,
  uploadDocument,
} from "@/api/knowledge";
import { getApiErrorMessage } from "@/api/http";
import { useKnowledgeChat } from "@/composables/useKnowledgeChat";
import type {
  DocumentRecord,
  KnowledgeStats,
} from "@/types/knowledge";

type ViewKey = "chat" | "documents";

const activeView = ref<ViewKey>("chat");
const darkMode = ref(false);
const documents = ref<DocumentRecord[]>([]);
const selectedDocumentId = ref<number>();
const documentLoading = ref(false);
const uploadBusy = ref(false);
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

async function refreshDocuments(): Promise<void> {
  documentLoading.value = true;

  try {
    documents.value = await listDocuments();

    if (
      selectedDocumentId.value &&
      !documents.value.some(
        (document) =>
          document.id === selectedDocumentId.value,
      )
    ) {
      selectedDocumentId.value = undefined;
    }

    await refreshChunkTotals();
  } catch (error) {
    showNotice("error", getApiErrorMessage(error));
  } finally {
    documentLoading.value = false;
  }
}

async function refreshChunkTotals(): Promise<void> {
  const settled = await Promise.allSettled(
    documents.value.map(async (document) => ({
      documentId: document.id,
      summary: await getChunkSummary(document.id),
    })),
  );

  const totals: Record<number, number> = {};

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
  uploadBusy.value = true;

  try {
    const document = await uploadDocument(file);
    busyDocumentId.value = document.id;

    showNotice(
      "success",
      `“${file.name}”上传成功，开始处理。`,
    );

    await processDocument(document.id);
    await createDocumentEmbeddings(document.id);
    await refreshDocuments();

    showNotice(
      "success",
      `“${file.name}”已完成知识加工。`,
    );
  } catch (error) {
    showNotice("error", getApiErrorMessage(error));
    await refreshDocuments();
  } finally {
    uploadBusy.value = false;
    busyDocumentId.value = undefined;
  }
}

async function handleProcess(
  documentId: number,
): Promise<void> {
  busyDocumentId.value = documentId;

  try {
    await processDocument(documentId);
    await refreshDocuments();
    showNotice("success", "文档处理完成。");
  } catch (error) {
    showNotice("error", getApiErrorMessage(error));
  } finally {
    busyDocumentId.value = undefined;
  }
}

async function handleEmbed(
  documentId: number,
): Promise<void> {
  busyDocumentId.value = documentId;

  try {
    const result =
      await createDocumentEmbeddings(documentId);
    await refreshDocuments();
    showNotice(
      "success",
      `向量化完成，共处理 ${result.processed_count} 个分块。`,
    );
  } catch (error) {
    showNotice("error", getApiErrorMessage(error));
  } finally {
    busyDocumentId.value = undefined;
  }
}

async function handleDelete(
  document: DocumentRecord,
): Promise<void> {
  if (
    !window.confirm(
      `确认删除“${document.filename}”吗？`,
    )
  ) {
    return;
  }

  busyDocumentId.value = document.id;

  try {
    await deleteDocument(document.id);
    await refreshDocuments();
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
        :loading="documentLoading"
        :busy-document-id="busyDocumentId"
        :upload-busy="uploadBusy"
        @refresh="refreshDocuments"
        @upload="handleUpload"
        @process="handleProcess"
        @embed="handleEmbed"
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
