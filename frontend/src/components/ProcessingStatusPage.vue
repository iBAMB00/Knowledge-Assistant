<script setup lang="ts">
import { CheckCircle2, CircleAlert, FileText, LoaderCircle, RefreshCw } from "lucide-vue-next";
import type { DocumentRecord, KnowledgeBaseRecord, ProcessingJobSnapshot, ProcessingJobStage } from "@/types/knowledge";

const props = defineProps<{
  knowledgeBase?: KnowledgeBaseRecord;
  documents: DocumentRecord[];
  jobsByDocumentId: Record<number, ProcessingJobSnapshot>;
  loading: boolean;
}>();

const emit = defineEmits<{ refresh: [] }>();

function jobFor(document: DocumentRecord) {
  return props.jobsByDocumentId[document.id] ?? document.active_job ?? null;
}

function stageLabel(stage?: ProcessingJobStage): string {
  const labels: Record<ProcessingJobStage, string> = {
    queued: "等待处理", parsing: "正在解析文档", chunking: "正在生成切片",
    embedding: "正在生成向量", indexing: "正在同步索引", finalizing: "正在完成处理", completed: "处理完成",
  };
  return stage ? labels[stage] : "暂无任务";
}

function statusLabel(status: string): string {
  const labels: Record<string, string> = { pending: "等待中", running: "处理中", succeeded: "已完成", failed: "失败" };
  return labels[status] ?? status;
}

function formatDate(value?: string | null): string {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat("zh-CN", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" }).format(date);
}
</script>

<template>
  <section class="page-shell">
    <header class="page-heading-row">
      <div>
        <p class="eyebrow">Processing Jobs</p>
        <h1>文档处理状态</h1>
        <p>{{ knowledgeBase ? `查看「${knowledgeBase.name}」文档解析、切片、向量化和索引状态。` : '请先选择一个知识库。' }}</p>
      </div>
      <button type="button" class="secondary-button" :disabled="loading" @click="emit('refresh')"><RefreshCw :size="16" :class="{ spinning: loading }" />刷新状态</button>
    </header>

    <div class="table-card processing-table">
      <div class="table-head processing-grid"><span>文件名</span><span>任务状态</span><span>进度</span><span>详情</span><span>更新时间</span></div>
      <div v-if="!knowledgeBase" class="empty-state"><FileText :size="38" /><p>请先在知识库页面选择一个知识库。</p></div>
      <div v-else-if="loading && documents.length === 0" class="empty-state"><LoaderCircle :size="28" class="spinning" /><p>正在读取任务状态…</p></div>
      <div v-else-if="documents.length === 0" class="empty-state"><FileText :size="38" /><p>当前知识库暂无文档。</p></div>

      <article v-for="document in documents" :key="document.id" class="table-row processing-grid">
        <div class="document-name-cell"><span class="table-leading-icon"><FileText :size="18" /></span><div><strong>{{ document.filename }}</strong><small>Document #{{ document.id }}</small></div></div>
        <span v-if="jobFor(document)" class="status-pill" :class="`job-pill-${jobFor(document)?.status}`">
          <LoaderCircle v-if="jobFor(document)?.status === 'pending' || jobFor(document)?.status === 'running'" :size="13" class="spinning" />
          <CircleAlert v-else-if="jobFor(document)?.status === 'failed'" :size="13" />
          <CheckCircle2 v-else :size="13" />
          {{ statusLabel(jobFor(document)?.status ?? '') }}
        </span>
        <span v-else class="muted">暂无任务</span>
        <div class="progress-inline"><div class="progress-track mini"><i :class="`job-${jobFor(document)?.status}`" :style="{ width: `${jobFor(document)?.progress ?? 0}%` }" /></div><strong>{{ jobFor(document)?.progress ?? 0 }}%</strong></div>
        <div><strong class="small-text">{{ stageLabel(jobFor(document)?.stage) }}</strong><small v-if="jobFor(document)?.error_message" class="danger-text block">{{ jobFor(document)?.error_message }}</small></div>
        <span class="muted small-text">{{ formatDate(jobFor(document)?.updated_at || document.created_at) }}</span>
      </article>
    </div>
  </section>
</template>
