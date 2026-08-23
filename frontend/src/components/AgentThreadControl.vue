<script setup lang="ts">
import {
  Check,
  CirclePause,
  CircleStop,
  LoaderCircle,
  Play,
  RefreshCw,
  RotateCcw,
  ShieldCheck,
  X,
} from "lucide-vue-next";
import { computed } from "vue";
import type {
  AgentThreadAction,
  AgentThreadStatusResponse,
} from "@/types/knowledge";

const props = defineProps<{
  thread?: AgentThreadStatusResponse | null;
  loading: boolean;
  busyAction?: AgentThreadAction;
  error: string;
  submitting: boolean;
}>();

const emit = defineEmits<{
  refresh: [];
  approve: [];
  reject: [];
  resume: [];
  cancel: [];
}>();

const statusLabel = computed(() => {
  switch (props.thread?.status) {
    case "running": return "运行中";
    case "waiting": return "等待确认";
    case "succeeded": return "已完成";
    case "failed": return "失败";
    case "cancelled": return "已取消";
    case "ready": return "就绪";
    default: return "未创建";
  }
});

const statusIcon = computed(() => {
  switch (props.thread?.status) {
    case "running": return LoaderCircle;
    case "waiting": return CirclePause;
    case "cancelled": return CircleStop;
    case "succeeded": return Check;
    case "failed": return X;
    default: return ShieldCheck;
  }
});

const actionLocked = computed(() => Boolean(props.busyAction));

const approvalTitle = computed(() => {
  if (props.thread?.can_resume && !props.thread.can_approve) {
    return "已批准，等待继续执行";
  }
  return "等待人工确认";
});
</script>

<template>
  <section class="agent-thread-card" aria-label="Stateful Agent Thread">
    <div class="agent-thread-heading">
      <div>
        <strong><ShieldCheck :size="15" /> Stateful Thread</strong>
        <span>Checkpoint · Resume · HITL · Cancel</span>
      </div>
      <button
        v-if="thread"
        type="button"
        class="icon-button thread-refresh-button"
        :disabled="loading || actionLocked"
        title="刷新 Thread 状态"
        @click="emit('refresh')"
      >
        <RefreshCw :size="14" :class="{ spin: loading || busyAction === 'refresh' }" />
      </button>
    </div>

    <div v-if="loading && !thread" class="thread-empty-state">
      <LoaderCircle :size="15" class="spin" />
      正在读取 Thread 状态…
    </div>

    <template v-else-if="thread">
      <div class="thread-status-row">
        <span class="thread-status-badge" :class="`thread-status-${thread.status}`">
          <component :is="statusIcon" :size="13" :class="{ spin: thread.status === 'running' }" />
          {{ statusLabel }}
        </span>
        <span class="thread-retry">Recovery #{{ thread.retry_count }}</span>
      </div>

      <code class="thread-id">{{ thread.thread_id }}</code>

      <p v-if="thread.last_error_code" class="thread-error-code">
        状态码：{{ thread.last_error_code }}
      </p>

      <div v-if="thread.status === 'waiting' && thread.pending_approvals.length" class="approval-list">
        <div class="approval-title">
          <CirclePause :size="14" />
          <span>{{ approvalTitle }}</span>
        </div>
        <article
          v-for="approval in thread.pending_approvals"
          :key="approval.call_id"
          class="approval-item"
        >
          <strong>{{ approval.tool_name }}</strong>
          <p>{{ approval.reason }}</p>
          <small>Call · {{ approval.call_id }}</small>
        </article>
      </div>

      <div class="thread-actions">
        <button
          v-if="thread.can_approve"
          type="button"
          class="thread-action approve"
          :disabled="actionLocked || submitting"
          @click="emit('approve')"
        >
          <LoaderCircle v-if="busyAction === 'approve'" :size="14" class="spin" />
          <Check v-else :size="14" />
          批准
        </button>

        <button
          v-if="thread.can_reject"
          type="button"
          class="thread-action reject"
          :disabled="actionLocked || submitting"
          @click="emit('reject')"
        >
          <LoaderCircle v-if="busyAction === 'reject'" :size="14" class="spin" />
          <X v-else :size="14" />
          拒绝
        </button>

        <button
          v-if="thread.can_resume"
          type="button"
          class="thread-action resume"
          :disabled="actionLocked || submitting"
          @click="emit('resume')"
        >
          <LoaderCircle v-if="busyAction === 'resume'" :size="14" class="spin" />
          <Play v-else :size="14" />
          继续执行
        </button>

        <button
          v-if="thread.can_cancel"
          type="button"
          class="thread-action cancel"
          :disabled="actionLocked"
          @click="emit('cancel')"
        >
          <LoaderCircle v-if="busyAction === 'cancel'" :size="14" class="spin" />
          <CircleStop v-else :size="14" />
          取消任务
        </button>
      </div>
    </template>

    <div v-else class="thread-empty-state">
      <RotateCcw :size="15" />
      发送第一条 LangGraph 消息后创建 durable Thread。
    </div>

    <p v-if="error" class="runtime-warning thread-error">{{ error }}</p>
  </section>
</template>
