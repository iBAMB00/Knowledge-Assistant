<script setup lang="ts">
import {
  Bot,
  CircleHelp,
  Database,
  FileText,
  MessageCircle,
  Settings,
  ShieldCheck,
} from "lucide-vue-next";
import type {
  KnowledgeStats,
} from "@/types/knowledge";

type ViewKey = "chat" | "documents";

defineProps<{
  activeView: ViewKey;
  stats: KnowledgeStats;
}>();

const emit = defineEmits<{
  navigate: [view: ViewKey];
}>();

function formatDate(value?: string): string {
  if (!value) {
    return "暂无";
  }

  const date = new Date(value);

  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}
</script>

<template>
  <aside class="sidebar">
    <div class="brand">
      <div class="brand-mark">
        <ShieldCheck :size="30" />
      </div>
      <div>
        <h1>Knowledge Assistant</h1>
        <p>您的智能知识助手</p>
      </div>
    </div>

    <nav class="primary-navigation">
      <button
        type="button"
        class="nav-item"
        :class="{ active: activeView === 'chat' }"
        @click="emit('navigate', 'chat')"
      >
        <MessageCircle :size="19" />
        <span>智能对话</span>
      </button>

      <button
        type="button"
        class="nav-item"
        :class="{
          active: activeView === 'documents',
        }"
        @click="emit('navigate', 'documents')"
      >
        <Database :size="19" />
        <span>知识库管理</span>
      </button>

      <button
        type="button"
        class="nav-item nav-item-muted"
        title="MVP 暂未开放独立页面"
      >
        <FileText :size="19" />
        <span>文档管理</span>
      </button>

      <button
        type="button"
        class="nav-item nav-item-muted"
        title="MVP 暂未开放"
      >
        <Settings :size="19" />
        <span>系统设置</span>
      </button>

      <button
        type="button"
        class="nav-item nav-item-muted"
        title="MVP 暂未开放"
      >
        <CircleHelp :size="19" />
        <span>使用帮助</span>
      </button>
    </nav>

    <section class="stats-card">
      <h2>知识库统计</h2>
      <dl>
        <div>
          <dt>文档总数</dt>
          <dd>{{ stats.documentCount }}</dd>
        </div>
        <div>
          <dt>分块总数</dt>
          <dd>{{ stats.chunkCount }}</dd>
        </div>
        <div>
          <dt>向量分块</dt>
          <dd>{{ stats.vectorChunkCount }}</dd>
        </div>
        <div class="stats-date">
          <dt>最近更新</dt>
          <dd>{{ formatDate(stats.lastUpdated) }}</dd>
        </div>
      </dl>
    </section>

    <div class="sidebar-user">
      <div class="avatar">
        <Bot :size="22" />
      </div>
      <div class="user-copy">
        <strong>admin</strong>
        <span><i />在线</span>
      </div>
    </div>
  </aside>
</template>
