<script setup lang="ts">
import { Moon, Sun, UserRound } from "lucide-vue-next";
import type { UserRecord } from "@/types/knowledge";

defineProps<{
  user: UserRecord;
  darkMode: boolean;
}>();

const emit = defineEmits<{ toggleTheme: [] }>();

function formatDate(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat("zh-CN", { dateStyle: "long", timeStyle: "short" }).format(date);
}
</script>

<template>
  <section class="page-shell narrow-page">
    <header class="page-heading-row"><div><p class="eyebrow">Account</p><h1>个人设置</h1><p>查看当前账户信息与界面偏好。</p></div></header>

    <div class="settings-card profile-card">
      <div class="profile-avatar"><UserRound :size="28" /></div>
      <div class="profile-title"><strong>{{ user.email }}</strong><span>{{ user.role === 'admin' ? '管理员' : '普通用户' }}</span></div>

      <div class="read-only-grid profile-grid">
        <div><span>邮箱</span><strong>{{ user.email }}</strong></div>
        <div><span>用户 ID</span><strong>{{ user.id }}</strong></div>
        <div><span>角色</span><strong>{{ user.role }}</strong></div>
        <div><span>账户状态</span><strong>{{ user.is_active ? '正常' : '已停用' }}</strong></div>
        <div class="span-two"><span>创建时间</span><strong>{{ formatDate(user.created_at) }}</strong></div>
      </div>

      <div class="preference-row">
        <div><strong>界面主题</strong><span>切换浅色或深色模式。</span></div>
        <button type="button" class="secondary-button" @click="emit('toggleTheme')"><Sun v-if="darkMode" :size="16" /><Moon v-else :size="16" />{{ darkMode ? '切换浅色' : '切换深色' }}</button>
      </div>
      <div class="preference-row"><div><strong>语言</strong><span>当前界面语言。</span></div><span class="status-pill neutral-pill">简体中文</span></div>
    </div>
  </section>
</template>
