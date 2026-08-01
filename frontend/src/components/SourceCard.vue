<script setup lang="ts">
import { FileText } from "lucide-vue-next";
import type {
  DocumentRecord,
  KnowledgeChatSource,
} from "@/types/knowledge";

const props = defineProps<{
  source: KnowledgeChatSource;
  documents: DocumentRecord[];
}>();

function filename(): string {
  return (
    props.documents.find(
      (item) =>
        item.id === props.source.document_id,
    )?.filename ??
    `文档 #${props.source.document_id}`
  );
}
</script>

<template>
  <article class="source-card">
    <div class="source-icon">
      <FileText :size="18" />
    </div>
    <div class="source-main">
      <strong>{{ filename() }}</strong>
      <span>来源 {{ source.source_number }}</span>
      <p>{{ source.excerpt }}</p>
    </div>
  </article>
</template>
