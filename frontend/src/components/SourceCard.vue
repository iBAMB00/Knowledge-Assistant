<script setup lang="ts">
import { FileText, Layers3, MapPin } from "lucide-vue-next";
import type { KnowledgeChatSource } from "@/types/knowledge";

defineProps<{ source: KnowledgeChatSource }>();

function pageLabel(source: KnowledgeChatSource): string {
  if (source.page_numbers.length > 0) return `第 ${source.page_numbers.join("、")} 页`;
  if (source.start_page && source.end_page && source.start_page !== source.end_page) {
    return `第 ${source.start_page}-${source.end_page} 页`;
  }
  if (source.start_page) return `第 ${source.start_page} 页`;
  return "无页码信息";
}
</script>

<template>
  <article class="source-card">
    <div class="source-card-heading">
      <span class="source-number">{{ source.source_number }}</span>
      <div>
        <strong><FileText :size="14" /> {{ source.filename }}</strong>
        <span>Chunk #{{ source.chunk_id }} · Document #{{ source.document_id }}</span>
      </div>
    </div>

    <div class="source-meta-row">
      <span><Layers3 :size="13" /> {{ source.section_title || source.heading_path.at(-1) || '未标注章节' }}</span>
      <span><MapPin :size="13" /> {{ pageLabel(source) }}</span>
    </div>

    <p>{{ source.excerpt }}</p>
  </article>
</template>
