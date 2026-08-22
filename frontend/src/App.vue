<script setup lang="ts">
import {
  CheckCircle2,
  Database,
  LoaderCircle,
  MessageCircle,
  RefreshCw,
  UserRound,
  XCircle,
} from "lucide-vue-next";
import { computed, onBeforeUnmount, onMounted, ref } from "vue";
import { getCurrentUser, loginUser, registerUser } from "@/api/auth";
import { clearAccessToken, getAccessToken, getApiErrorMessage, setAccessToken } from "@/api/http";
import {
  createKnowledgeBase,
  deleteKnowledgeBase,
  listKnowledgeBases,
  updateKnowledgeBase,
} from "@/api/knowledgeBases";
import {
  createProcessingJob,
  deleteDocument,
  listDocuments,
  uploadDocument,
} from "@/api/knowledge";
import AppSidebar from "@/components/AppSidebar.vue";
import ChatWorkspace from "@/components/ChatWorkspace.vue";
import DocumentManager from "@/components/DocumentManager.vue";
import KnowledgeBaseList from "@/components/KnowledgeBaseList.vue";
import KnowledgeBaseSettings from "@/components/KnowledgeBaseSettings.vue";
import LoginPage from "@/components/LoginPage.vue";
import ProcessingStatusPage from "@/components/ProcessingStatusPage.vue";
import ProfilePage from "@/components/ProfilePage.vue";
import { useAgentChat } from "@/composables/useAgentChat";
import { useKnowledgeChat } from "@/composables/useKnowledgeChat";
import { useProcessingJobPolling } from "@/composables/useProcessingJobPolling";
import type {
  AgentRuntime,
  AppView,
  ChatMode,
  DocumentRecord,
  KnowledgeBaseRecord,
  ProcessingJobSnapshot,
  UserRecord,
} from "@/types/knowledge";

const currentUser = ref<UserRecord | null>(null);
const authChecking = ref(true);
const authBusy = ref(false);
const authError = ref("");

const activeView = ref<AppView>("knowledge-bases");
const knowledgeBases = ref<KnowledgeBaseRecord[]>([]);
const knowledgeBaseLoading = ref(false);
const knowledgeBaseBusy = ref(false);
const documentCounts = ref<Record<number, number>>({});
const selectedKnowledgeBaseId = ref<number>();

const documents = ref<DocumentRecord[]>([]);
const documentLoading = ref(false);
const uploadBusy = ref(false);
const uploadProgress = ref<number | null>(null);
const busyDocumentId = ref<number>();
const selectedDocumentId = ref<number>();

const chatMode = ref<ChatMode>("knowledge");
const darkMode = ref(false);
const notice = ref<{ type: "success" | "error"; message: string } | null>(null);
let noticeTimer: number | undefined;

const selectedKnowledgeBase = computed(() =>
  knowledgeBases.value.find((kb) => kb.id === selectedKnowledgeBaseId.value),
);

const {
  jobsByDocumentId,
  syncDocuments,
  trackJob,
  forgetDocument,
} = useProcessingJobPolling({
  onTerminalJobs: handleTerminalJobs,
  onPollError: () => showNotice("error", "任务状态刷新暂时失败，系统会自动重试。"),
});

const {
  messages: knowledgeMessages,
  submitting: knowledgeSubmitting,
  streamingEnabled: knowledgeStreamingEnabled,
  sendQuestion: sendKnowledgeQuestion,
  stopGeneration: stopKnowledgeGeneration,
  clearConversation: clearKnowledgeConversation,
} = useKnowledgeChat();

const {
  messages: agentMessages,
  submitting: agentSubmitting,
  streamingEnabled: agentStreamingEnabled,
  selectedRuntime: agentRuntime,
  runtimeOptions: agentRuntimeOptions,
  runtimeLoading: agentRuntimeLoading,
  runtimeError: agentRuntimeError,
  loadRuntimes: loadAgentRuntimes,
  setRuntime: setAgentRuntime,
  sendQuestion: sendAgentQuestion,
  stopGeneration: stopAgentGeneration,
  clearConversation: clearAgentConversation,
  resetRuntimeState: resetAgentRuntimeState,
} = useAgentChat();

onMounted(async () => {
  darkMode.value = localStorage.getItem("knowledge-assistant-theme") === "dark";
  applyTheme();
  window.addEventListener("knowledge-assistant:unauthorized", handleUnauthorized);

  try {
    if (!getAccessToken()) return;
    currentUser.value = await getCurrentUser();
    await loadKnowledgeBases(true);
  } catch {
    clearSession();
  } finally {
    authChecking.value = false;
  }
});

onBeforeUnmount(() => {
  window.removeEventListener("knowledge-assistant:unauthorized", handleUnauthorized);
  if (noticeTimer) window.clearTimeout(noticeTimer);
});

async function handleLogin(email: string, password: string): Promise<void> {
  authBusy.value = true;
  authError.value = "";
  try {
    const token = await loginUser(email, password);
    setAccessToken(token.access_token);
    currentUser.value = await getCurrentUser();
    activeView.value = "knowledge-bases";
    await loadKnowledgeBases(true);
  } catch (error) {
    clearAccessToken();
    authError.value = getApiErrorMessage(error);
  } finally {
    authBusy.value = false;
  }
}

async function handleRegister(email: string, password: string): Promise<void> {
  authBusy.value = true;
  authError.value = "";
  try {
    await registerUser(email, password);
    const token = await loginUser(email, password);
    setAccessToken(token.access_token);
    currentUser.value = await getCurrentUser();
    activeView.value = "knowledge-bases";
    await loadKnowledgeBases(true);
    showNotice("success", "账号创建成功，欢迎使用 Knowledge Assistant。" );
  } catch (error) {
    clearAccessToken();
    authError.value = getApiErrorMessage(error);
  } finally {
    authBusy.value = false;
  }
}

function handleUnauthorized(): void {
  if (!currentUser.value) return;
  clearSession();
  authError.value = "登录状态已失效，请重新登录。";
}

function logout(): void {
  clearSession();
  authError.value = "";
}

function clearSession(): void {
  stopKnowledgeGeneration();
  stopAgentGeneration();
  clearAccessToken();
  currentUser.value = null;
  knowledgeBases.value = [];
  documents.value = [];
  documentCounts.value = {};
  selectedKnowledgeBaseId.value = undefined;
  selectedDocumentId.value = undefined;
  syncDocuments([]);
  clearKnowledgeConversation();
  clearAgentConversation();
  resetAgentRuntimeState();
  chatMode.value = "knowledge";
}

async function loadKnowledgeBases(loadCounts = false): Promise<void> {
  knowledgeBaseLoading.value = true;
  try {
    knowledgeBases.value = await listKnowledgeBases();

    if (
      selectedKnowledgeBaseId.value &&
      !knowledgeBases.value.some((kb) => kb.id === selectedKnowledgeBaseId.value)
    ) {
      selectedKnowledgeBaseId.value = undefined;
      documents.value = [];
      syncDocuments([]);
    }

    if (loadCounts) {
      const settled = await Promise.allSettled(
        knowledgeBases.value.map(async (kb) => ({
          id: kb.id,
          docs: await listDocuments(kb.id),
        })),
      );
      const counts: Record<number, number> = {};
      for (const result of settled) {
        if (result.status === "fulfilled") counts[result.value.id] = result.value.docs.length;
      }
      documentCounts.value = counts;
    }
  } catch (error) {
    showNotice("error", getApiErrorMessage(error));
  } finally {
    knowledgeBaseLoading.value = false;
  }
}

async function refreshDocuments(showLoading = true): Promise<void> {
  const knowledgeBaseId = selectedKnowledgeBaseId.value;
  if (!knowledgeBaseId) {
    documents.value = [];
    syncDocuments([]);
    return;
  }

  if (showLoading) documentLoading.value = true;
  try {
    documents.value = await listDocuments(knowledgeBaseId);
    syncDocuments(documents.value);
    documentCounts.value = {
      ...documentCounts.value,
      [knowledgeBaseId]: documents.value.length,
    };

    if (
      selectedDocumentId.value &&
      !documents.value.some((document) => document.id === selectedDocumentId.value)
    ) {
      selectedDocumentId.value = undefined;
    }
  } catch (error) {
    showNotice("error", getApiErrorMessage(error));
  } finally {
    if (showLoading) documentLoading.value = false;
  }
}

async function openKnowledgeBase(kb: KnowledgeBaseRecord): Promise<void> {
  selectedKnowledgeBaseId.value = kb.id;
  selectedDocumentId.value = undefined;
  activeView.value = "documents";
  await refreshDocuments();
}

async function navigate(view: AppView): Promise<void> {
  activeView.value = view;

  if (view === "knowledge-bases") {
    await loadKnowledgeBases(false);
    return;
  }

  if (["chat", "processing", "documents", "knowledge-base-settings"].includes(view)) {
    if (!selectedKnowledgeBaseId.value && knowledgeBases.value.length > 0) {
      selectedKnowledgeBaseId.value = knowledgeBases.value[0].id;
    }
    await refreshDocuments(false);
    if (view === "chat") void loadAgentRuntimes();
  }
}

async function handleCreateKnowledgeBase(name: string, description: string | null): Promise<void> {
  knowledgeBaseBusy.value = true;
  try {
    const created = await createKnowledgeBase(name, description);
    await loadKnowledgeBases(true);
    selectedKnowledgeBaseId.value = created.id;
    showNotice("success", `知识库“${created.name}”已创建。`);
  } catch (error) {
    showNotice("error", getApiErrorMessage(error));
  } finally {
    knowledgeBaseBusy.value = false;
  }
}

async function handleUpdateKnowledgeBase(
  id: number,
  name: string,
  description: string | null,
): Promise<void> {
  knowledgeBaseBusy.value = true;
  try {
    const updated = await updateKnowledgeBase(id, { name, description });
    knowledgeBases.value = knowledgeBases.value.map((kb) => (kb.id === id ? updated : kb));
    showNotice("success", "知识库设置已保存。" );
  } catch (error) {
    showNotice("error", getApiErrorMessage(error));
  } finally {
    knowledgeBaseBusy.value = false;
  }
}

async function handleRemoveKnowledgeBase(kb: KnowledgeBaseRecord): Promise<void> {
  if (!window.confirm(`确定删除知识库“${kb.name}”吗？存在文档时后端会拒绝删除。`)) return;
  knowledgeBaseBusy.value = true;
  try {
    await deleteKnowledgeBase(kb.id);
    if (selectedKnowledgeBaseId.value === kb.id) {
      selectedKnowledgeBaseId.value = undefined;
      documents.value = [];
      syncDocuments([]);
      activeView.value = "knowledge-bases";
    }
    await loadKnowledgeBases(true);
    showNotice("success", `知识库“${kb.name}”已删除。`);
  } catch (error) {
    showNotice("error", getApiErrorMessage(error));
  } finally {
    knowledgeBaseBusy.value = false;
  }
}

async function handleUpload(file: File): Promise<void> {
  const knowledgeBaseId = selectedKnowledgeBaseId.value;
  if (!knowledgeBaseId || uploadBusy.value) return;

  uploadBusy.value = true;
  uploadProgress.value = 0;
  try {
    const uploaded = await uploadDocument(knowledgeBaseId, file, (progress) => {
      uploadProgress.value = progress;
    });
    await refreshDocuments(false);
    showNotice("success", `“${uploaded.filename}”上传完成，正在创建后台处理任务。`);

    try {
      const job = await createProcessingJob(uploaded.id, "full_pipeline");
      trackJob(job);
      await refreshDocuments(false);
    } catch (error) {
      showNotice("error", `文件已上传，但处理任务创建失败：${getApiErrorMessage(error)}`);
    }
  } catch (error) {
    showNotice("error", getApiErrorMessage(error));
  } finally {
    uploadBusy.value = false;
    uploadProgress.value = null;
  }
}

async function handleStartProcessing(documentId: number): Promise<void> {
  if (busyDocumentId.value) return;
  busyDocumentId.value = documentId;
  try {
    const job = await createProcessingJob(documentId, "full_pipeline");
    trackJob(job);
    await refreshDocuments(false);
    showNotice("success", `文档 #${documentId} 的处理任务已提交。`);
  } catch (error) {
    showNotice("error", getApiErrorMessage(error));
  } finally {
    busyDocumentId.value = undefined;
  }
}

async function handleRemoveDocument(document: DocumentRecord): Promise<void> {
  if (!window.confirm(`确定删除“${document.filename}”吗？该操作会触发后端文档清理语义。`)) return;
  busyDocumentId.value = document.id;
  try {
    await deleteDocument(document.id);
    forgetDocument(document.id);
    if (selectedDocumentId.value === document.id) selectedDocumentId.value = undefined;
    await refreshDocuments(false);
    showNotice("success", `“${document.filename}”已删除。`);
  } catch (error) {
    showNotice("error", getApiErrorMessage(error));
  } finally {
    busyDocumentId.value = undefined;
  }
}

async function handleTerminalJobs(_jobs: ProcessingJobSnapshot[]): Promise<void> {
  await refreshDocuments(false);
}

async function handleChatKnowledgeBaseChange(id?: number): Promise<void> {
  if (selectedKnowledgeBaseId.value === id) return;
  selectedKnowledgeBaseId.value = id;
  selectedDocumentId.value = undefined;
  clearKnowledgeConversation();
  clearAgentConversation();
  await refreshDocuments(false);
}

async function handleSendQuestion(question: string): Promise<void> {
  const knowledgeBaseId = selectedKnowledgeBaseId.value;
  if (!knowledgeBaseId) {
    showNotice("error", "请先选择一个知识库。" );
    return;
  }
  await sendKnowledgeQuestion(question, knowledgeBaseId, selectedDocumentId.value);
}

function handleDetailTab(view: "documents" | "chat" | "knowledge-base-settings"): void {
  activeView.value = view;
  if (view === "chat") void loadAgentRuntimes();
}

function handleSelectedDocumentChange(id?: number): void {
  selectedDocumentId.value = id;
}

function handleKnowledgeStreamingEnabledChange(enabled: boolean): void {
  knowledgeStreamingEnabled.value = enabled;
}

function handleAgentStreamingEnabledChange(enabled: boolean): void {
  agentStreamingEnabled.value = enabled;
}

function handleChatModeChange(mode: ChatMode): void {
  chatMode.value = mode;
  if (mode === "agent") void loadAgentRuntimes();
}

function handleAgentRuntimeChange(runtime: AgentRuntime): void {
  setAgentRuntime(runtime);
}

async function handleSendAgentQuestion(question: string): Promise<void> {
  const knowledgeBaseId = selectedKnowledgeBaseId.value;
  if (!knowledgeBaseId) {
    showNotice("error", "请先选择一个知识库作为 Agent 的可信执行范围。");
    return;
  }
  await sendAgentQuestion(question, knowledgeBaseId);
}

async function handleUpdateSelectedKnowledgeBase(
  name: string,
  description: string | null,
): Promise<void> {
  const knowledgeBase = selectedKnowledgeBase.value;
  if (!knowledgeBase) return;
  await handleUpdateKnowledgeBase(knowledgeBase.id, name, description);
}

async function handleRemoveSelectedKnowledgeBase(): Promise<void> {
  const knowledgeBase = selectedKnowledgeBase.value;
  if (!knowledgeBase) return;
  await handleRemoveKnowledgeBase(knowledgeBase);
}

function toggleTheme(): void {
  darkMode.value = !darkMode.value;
  localStorage.setItem("knowledge-assistant-theme", darkMode.value ? "dark" : "light");
  applyTheme();
}

function applyTheme(): void {
  document.documentElement.dataset.theme = darkMode.value ? "dark" : "light";
}

function showNotice(type: "success" | "error", message: string): void {
  notice.value = { type, message };
  if (noticeTimer) window.clearTimeout(noticeTimer);
  noticeTimer = window.setTimeout(() => {
    notice.value = null;
  }, 4200);
}
</script>

<template>
  <div v-if="authChecking" class="boot-screen">
    <LoaderCircle :size="34" class="spinning" />
    <strong>Knowledge Assistant</strong>
    <span>正在恢复登录状态…</span>
  </div>

  <LoginPage
    v-else-if="!currentUser"
    :busy="authBusy"
    :error="authError"
    @login="handleLogin"
    @register="handleRegister"
  />

  <div v-else class="app-shell">
    <AppSidebar
      :active-view="activeView"
      :user="currentUser"
      @navigate="navigate"
      @logout="logout"
    />

    <main class="app-main">
      <KnowledgeBaseList
        v-if="activeView === 'knowledge-bases'"
        :knowledge-bases="knowledgeBases"
        :document-counts="documentCounts"
        :loading="knowledgeBaseLoading"
        :busy="knowledgeBaseBusy"
        @open="openKnowledgeBase"
        @create="handleCreateKnowledgeBase"
        @update="handleUpdateKnowledgeBase"
        @remove="handleRemoveKnowledgeBase"
        @refresh="loadKnowledgeBases(true)"
      />

      <DocumentManager
        v-else-if="activeView === 'documents' && selectedKnowledgeBase"
        :knowledge-base="selectedKnowledgeBase"
        :documents="documents"
        :jobs-by-document-id="jobsByDocumentId"
        :loading="documentLoading"
        :busy-document-id="busyDocumentId"
        :upload-busy="uploadBusy"
        :upload-progress="uploadProgress"
        @refresh="refreshDocuments()"
        @upload="handleUpload"
        @start="handleStartProcessing"
        @remove="handleRemoveDocument"
        @tab="handleDetailTab"
      />

      <ProcessingStatusPage
        v-else-if="activeView === 'processing'"
        :knowledge-base="selectedKnowledgeBase"
        :documents="documents"
        :jobs-by-document-id="jobsByDocumentId"
        :loading="documentLoading"
        @refresh="refreshDocuments()"
      />

      <ChatWorkspace
        v-else-if="activeView === 'chat'"
        :mode="chatMode"
        :knowledge-bases="knowledgeBases"
        :selected-knowledge-base-id="selectedKnowledgeBaseId"
        :documents="documents"
        :selected-document-id="selectedDocumentId"
        :knowledge-messages="knowledgeMessages"
        :knowledge-submitting="knowledgeSubmitting"
        :knowledge-streaming-enabled="knowledgeStreamingEnabled"
        :agent-messages="agentMessages"
        :agent-submitting="agentSubmitting"
        :agent-streaming-enabled="agentStreamingEnabled"
        :agent-runtime="agentRuntime"
        :agent-runtime-options="agentRuntimeOptions"
        :agent-runtime-loading="agentRuntimeLoading"
        :agent-runtime-error="agentRuntimeError"
        @update:mode="handleChatModeChange"
        @update:selected-knowledge-base-id="handleChatKnowledgeBaseChange"
        @update:selected-document-id="handleSelectedDocumentChange"
        @update:knowledge-streaming-enabled="handleKnowledgeStreamingEnabledChange"
        @update:agent-streaming-enabled="handleAgentStreamingEnabledChange"
        @update:agent-runtime="handleAgentRuntimeChange"
        @send-knowledge="handleSendQuestion"
        @send-agent="handleSendAgentQuestion"
        @stop-knowledge="stopKnowledgeGeneration"
        @stop-agent="stopAgentGeneration"
        @clear-knowledge="clearKnowledgeConversation"
        @clear-agent="clearAgentConversation"
      />

      <KnowledgeBaseSettings
        v-else-if="activeView === 'knowledge-base-settings' && selectedKnowledgeBase"
        :knowledge-base="selectedKnowledgeBase"
        :busy="knowledgeBaseBusy"
        @update="handleUpdateSelectedKnowledgeBase"
        @remove="handleRemoveSelectedKnowledgeBase"
        @tab="handleDetailTab"
      />

      <ProfilePage
        v-else-if="activeView === 'profile'"
        :user="currentUser"
        :dark-mode="darkMode"
        @toggle-theme="toggleTheme"
      />

      <section v-else class="page-shell">
        <div class="empty-state large"><Database :size="42" /><p>请先选择一个知识库。</p><button type="button" class="primary-button" @click="navigate('knowledge-bases')">返回知识库</button></div>
      </section>
    </main>

    <nav class="mobile-nav">
      <button :class="{ active: activeView === 'knowledge-bases' || activeView === 'documents' }" @click="navigate('knowledge-bases')"><Database :size="18" /><span>知识库</span></button>
      <button :class="{ active: activeView === 'chat' }" @click="navigate('chat')"><MessageCircle :size="18" /><span>助手</span></button>
      <button :class="{ active: activeView === 'processing' }" @click="navigate('processing')"><RefreshCw :size="18" /><span>处理</span></button>
      <button :class="{ active: activeView === 'profile' }" @click="navigate('profile')"><UserRound :size="18" /><span>我的</span></button>
    </nav>
  </div>

  <transition name="notice">
    <div v-if="notice" class="toast" :class="`toast-${notice.type}`">
      <CheckCircle2 v-if="notice.type === 'success'" :size="18" />
      <XCircle v-else :size="18" />
      <span>{{ notice.message }}</span>
    </div>
  </transition>
</template>
