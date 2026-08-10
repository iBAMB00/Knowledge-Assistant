<script setup lang="ts">
import {
  BookOpen,
  Bot,
  CircleUserRound,
  Database,
  LogOut,
  MessageCircle,
  RefreshCw,
} from "lucide-vue-next";
import type { AppView, UserRecord } from "@/types/knowledge";

defineProps<{
  activeView: AppView;
  user: UserRecord;
}>();

const emit = defineEmits<{
  navigate: [view: AppView];
  logout: [];
}>();
</script>

<template>
  <aside class="sidebar">
    <div class="brand">
      <span class="brand-icon"><BookOpen :size="25" /></span>
      <div>
        <strong>知识库助手</strong>
        <small>Knowledge Assistant</small>
      </div>
    </div>

    <nav class="sidebar-nav">
      <button
        type="button"
        class="nav-item"
        :class="{ active: activeView === 'knowledge-bases' || activeView === 'documents' || activeView === 'knowledge-base-settings' }"
        @click="emit('navigate', 'knowledge-bases')"
      >
        <Database :size="18" />
        <span>知识库</span>
      </button>

      <button
        type="button"
        class="nav-item"
        :class="{ active: activeView === 'chat' }"
        @click="emit('navigate', 'chat')"
      >
        <MessageCircle :size="18" />
        <span>聊天问答</span>
      </button>

      <button
        type="button"
        class="nav-item"
        :class="{ active: activeView === 'processing' }"
        @click="emit('navigate', 'processing')"
      >
        <RefreshCw :size="18" />
        <span>处理状态</span>
      </button>

      <button
        type="button"
        class="nav-item"
        :class="{ active: activeView === 'profile' }"
        @click="emit('navigate', 'profile')"
      >
        <CircleUserRound :size="18" />
        <span>个人设置</span>
      </button>
    </nav>

    <div class="sidebar-spacer" />

    <section class="sidebar-tip">
      <Bot :size="18" />
      <div>
        <strong>RAG 工作区</strong>
        <span>回答基于您有权访问的私有知识。</span>
      </div>
    </section>

    <footer class="sidebar-user">
      <div class="avatar">{{ user.email.slice(0, 1).toUpperCase() }}</div>
      <div class="sidebar-user-copy">
        <strong>{{ user.email }}</strong>
        <span>{{ user.role === 'admin' ? '管理员' : '普通用户' }}</span>
      </div>
      <button type="button" class="icon-button" title="退出登录" @click="emit('logout')">
        <LogOut :size="17" />
      </button>
    </footer>
  </aside>
</template>
