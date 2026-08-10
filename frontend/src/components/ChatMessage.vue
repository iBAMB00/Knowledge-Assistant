<script setup lang="ts">
import { Bot, Clock3, Copy, UserRound } from "lucide-vue-next";
import { ref } from "vue";
import SourceCard from "@/components/SourceCard.vue";
import type { ChatMessageRecord } from "@/types/knowledge";

const props = defineProps<{ message: ChatMessageRecord }>();
const copied = ref(false);

async function copyAnswer(): Promise<void> {
  if (!props.message.content) return;
  await navigator.clipboard.writeText(props.message.content);
  copied.value = true;
  window.setTimeout(() => (copied.value = false), 1500);
}

function elapsed(value?: number): string {
  if (value === undefined) return "";
  return value >= 1000 ? `${(value / 1000).toFixed(1)}s` : `${value}ms`;
}
</script>

<template>
  <div class="message-row" :class="`message-${message.role}`">
    <div class="message-avatar">
      <UserRound v-if="message.role === 'user'" :size="20" />
      <Bot v-else :size="21" />
    </div>

    <div class="message-column">
      <article class="message-bubble" :class="{ pending: message.pending, error: message.error }">
        <div v-if="message.content" class="message-content">{{ message.content }}</div>
        <div v-else-if="message.pending" class="typing-indicator"><span /><span /><span /></div>

        <section v-if="message.role === 'assistant' && message.sources.length" class="message-sources">
          <h3>参考来源</h3>
          <div class="source-list">
            <SourceCard v-for="source in message.sources" :key="`${source.document_id}-${source.chunk_id}-${source.source_number}`" :source="source" />
          </div>
        </section>
      </article>

      <div v-if="message.role === 'assistant' && !message.pending" class="message-meta">
        <span v-if="message.elapsedMs !== undefined"><Clock3 :size="13" /> {{ elapsed(message.elapsedMs) }}</span>
        <button v-if="message.content" type="button" @click="copyAnswer"><Copy :size="13" /> {{ copied ? '已复制' : '复制' }}</button>
      </div>
    </div>
  </div>
</template>
