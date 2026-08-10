<script setup lang="ts">
import { AlertTriangle, LoaderCircle, Save, Trash2 } from "lucide-vue-next";
import { ref, watch } from "vue";
import type { KnowledgeBaseRecord } from "@/types/knowledge";

const props = defineProps<{
  knowledgeBase: KnowledgeBaseRecord;
  busy: boolean;
}>();

const emit = defineEmits<{
  update: [name: string, description: string | null];
  remove: [];
  tab: [view: "documents" | "chat" | "knowledge-base-settings"];
}>();

const name = ref(props.knowledgeBase.name);
const description = ref(props.knowledgeBase.description ?? "");

watch(() => props.knowledgeBase, (kb) => {
  name.value = kb.name;
  description.value = kb.description ?? "";
});

function formatDate(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat("zh-CN", { dateStyle: "long", timeStyle: "short" }).format(date);
}
</script>

<template>
  <section class="page-shell">
    <div class="breadcrumb">知识库 <span>›</span> {{ knowledgeBase.name }} <span>›</span> 设置</div>
    <header class="kb-detail-heading compact-kb-heading">
      <div><h1>知识库设置</h1><p>管理知识库基本信息与访问边界。</p></div>
    </header>

    <nav class="detail-tabs">
      <button type="button" @click="emit('tab', 'documents')">文档管理</button>
      <button type="button" @click="emit('tab', 'chat')">聊天问答</button>
      <button type="button" class="active">知识库设置</button>
    </nav>

    <div class="settings-layout">
      <form class="settings-card" @submit.prevent="emit('update', name.trim(), description.trim() || null)">
        <div class="section-heading"><h2>基本信息</h2><p>名称和描述会显示在知识库列表与聊天选择器中。</p></div>
        <label class="form-field"><span>知识库名称</span><input v-model="name" maxlength="100" /></label>
        <label class="form-field"><span>知识库描述</span><textarea v-model="description" rows="5" maxlength="1000" /></label>
        <div class="read-only-grid">
          <div><span>Knowledge Base ID</span><strong>{{ knowledgeBase.id }}</strong></div>
          <div><span>Owner ID</span><strong>{{ knowledgeBase.owner_id }}</strong></div>
          <div><span>创建时间</span><strong>{{ formatDate(knowledgeBase.created_at) }}</strong></div>
          <div><span>更新时间</span><strong>{{ formatDate(knowledgeBase.updated_at) }}</strong></div>
        </div>
        <div class="settings-actions"><button type="submit" class="primary-button" :disabled="busy || !name.trim()"><LoaderCircle v-if="busy" :size="16" class="spinning" /><Save v-else :size="16" />保存设置</button></div>
      </form>

      <aside class="settings-card side-settings-card">
        <div class="section-heading"><h2>访问控制</h2><p>v1.0 使用 Ownership + RBAC 控制知识空间边界。</p></div>
        <div class="permission-summary"><strong>仅授权用户可见</strong><span>普通用户只能访问自己拥有的知识库；管理员按后端策略拥有管理能力。</span></div>
        <div class="danger-zone">
          <div><AlertTriangle :size="18" /><div><strong>删除知识库</strong><span>存在文档时后端可能拒绝删除，请先清理知识库内容。</span></div></div>
          <button type="button" class="danger-button" :disabled="busy" @click="emit('remove')"><Trash2 :size="15" />删除知识库</button>
        </div>
      </aside>
    </div>
  </section>
</template>
