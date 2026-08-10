<script setup lang="ts">
import {
  Database,
  Edit3,
  FolderOpen,
  LoaderCircle,
  Plus,
  Search,
  Trash2,
  X,
} from "lucide-vue-next";
import { computed, ref, watch } from "vue";
import type { KnowledgeBaseRecord } from "@/types/knowledge";

const props = defineProps<{
  knowledgeBases: KnowledgeBaseRecord[];
  documentCounts: Record<number, number>;
  loading: boolean;
  busy: boolean;
}>();

const emit = defineEmits<{
  open: [knowledgeBase: KnowledgeBaseRecord];
  create: [name: string, description: string | null];
  update: [id: number, name: string, description: string | null];
  remove: [knowledgeBase: KnowledgeBaseRecord];
  refresh: [];
}>();

const keyword = ref("");
const showEditor = ref(false);
const editingId = ref<number | null>(null);
const name = ref("");
const description = ref("");

const filtered = computed(() => {
  const q = keyword.value.trim().toLowerCase();
  if (!q) return props.knowledgeBases;
  return props.knowledgeBases.filter((kb) =>
    `${kb.name} ${kb.description ?? ""}`.toLowerCase().includes(q),
  );
});

watch(() => props.busy, (busy) => {
  if (!busy && showEditor.value && name.value.trim()) {
    // App 成功后会刷新列表；这里保留弹窗，由父组件通过 key 重建不是必须。
  }
});

function openCreate(): void {
  editingId.value = null;
  name.value = "";
  description.value = "";
  showEditor.value = true;
}

function openEdit(kb: KnowledgeBaseRecord): void {
  editingId.value = kb.id;
  name.value = kb.name;
  description.value = kb.description ?? "";
  showEditor.value = true;
}

function submit(): void {
  const normalized = name.value.trim();
  if (!normalized || props.busy) return;
  if (editingId.value) {
    emit("update", editingId.value, normalized, description.value.trim() || null);
  } else {
    emit("create", normalized, description.value.trim() || null);
  }
  showEditor.value = false;
}

function formatDate(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat("zh-CN", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}
</script>

<template>
  <section class="page-shell">
    <header class="page-heading-row">
      <div>
        <p class="eyebrow">Knowledge Bases</p>
        <h1>知识库</h1>
        <p>管理您的知识库，创建、查看和维护知识集合。</p>
      </div>
      <button type="button" class="primary-button" @click="openCreate">
        <Plus :size="17" /> 创建知识库
      </button>
    </header>

    <div class="toolbar-row">
      <label class="search-box wide">
        <Search :size="17" />
        <input v-model="keyword" type="search" placeholder="搜索知识库名称或描述..." />
      </label>
      <button type="button" class="secondary-button" :disabled="loading" @click="emit('refresh')">
        <LoaderCircle v-if="loading" :size="16" class="spinning" />
        <Database v-else :size="16" />
        刷新
      </button>
    </div>

    <div class="table-card kb-table-card">
      <div class="table-head kb-grid">
        <span>名称</span><span>描述</span><span>文档数量</span><span>更新时间</span><span>操作</span>
      </div>

      <div v-if="loading && knowledgeBases.length === 0" class="empty-state">
        <LoaderCircle :size="28" class="spinning" />
        <p>正在加载知识库…</p>
      </div>

      <div v-else-if="filtered.length === 0" class="empty-state">
        <FolderOpen :size="38" />
        <p>暂无知识库，创建一个开始使用。</p>
        <button type="button" class="primary-button compact" @click="openCreate">
          <Plus :size="16" /> 创建知识库
        </button>
      </div>

      <article v-for="kb in filtered" :key="kb.id" class="table-row kb-grid">
        <button type="button" class="kb-name-button" @click="emit('open', kb)">
          <span class="table-leading-icon"><Database :size="18" /></span>
          <strong>{{ kb.name }}</strong>
        </button>
        <span class="muted truncate">{{ kb.description || '暂无描述' }}</span>
        <span>{{ documentCounts[kb.id] ?? 0 }}</span>
        <span class="muted">{{ formatDate(kb.updated_at) }}</span>
        <div class="row-actions">
          <button type="button" class="text-action" @click="emit('open', kb)">查看</button>
          <button type="button" class="icon-button" title="编辑" @click="openEdit(kb)"><Edit3 :size="15" /></button>
          <button type="button" class="icon-button danger" title="删除" @click="emit('remove', kb)"><Trash2 :size="15" /></button>
        </div>
      </article>
    </div>

    <div v-if="showEditor" class="modal-backdrop" @click.self="showEditor = false">
      <form class="modal-card" @submit.prevent="submit">
        <div class="modal-heading">
          <div>
            <h2>{{ editingId ? '编辑知识库' : '创建知识库' }}</h2>
            <p>{{ editingId ? '修改知识库名称与描述。' : '创建一个新的私有知识空间。' }}</p>
          </div>
          <button type="button" class="icon-button" @click="showEditor = false"><X :size="18" /></button>
        </div>
        <label class="form-field"><span>知识库名称</span><input v-model="name" maxlength="100" placeholder="例如：产品文档库" /></label>
        <label class="form-field"><span>知识库描述</span><textarea v-model="description" maxlength="1000" rows="4" placeholder="描述这个知识库保存的内容" /></label>
        <div class="modal-actions">
          <button type="button" class="secondary-button" @click="showEditor = false">取消</button>
          <button type="submit" class="primary-button" :disabled="busy || !name.trim()">
            <LoaderCircle v-if="busy" :size="16" class="spinning" />
            {{ editingId ? '保存修改' : '创建知识库' }}
          </button>
        </div>
      </form>
    </div>
  </section>
</template>
