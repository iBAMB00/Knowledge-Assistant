<script setup lang="ts">
import {
  Bot,
  Clock3,
  Copy,
  UserRound,
} from "lucide-vue-next";
import { ref } from "vue";
import SourceCard from "@/components/SourceCard.vue";
import type {
  ChatMessageRecord,
  DocumentRecord,
} from "@/types/knowledge";

const props = defineProps<{
  message: ChatMessageRecord;
  documents: DocumentRecord[];
}>();

const copied = ref(false);

async function copyAnswer(): Promise<void> {
  if (!props.message.content) {
    return;
  }

  await navigator.clipboard.writeText(
    props.message.content,
  );
  copied.value = true;

  window.setTimeout(() => {
    copied.value = false;
  }, 1500);
}

function elapsed(value?: number): string {
  if (value === undefined) {
    return "";
  }

  return value >= 1000
    ? `${(value / 1000).toFixed(1)}s`
    : `${value}ms`;
}
</script>

<template>
  <div
    class="message-row"
    :class="`message-${message.role}`"
  >
    <div class="message-avatar">
      <UserRound
        v-if="message.role === 'user'"
        :size="21"
      />
      <Bot v-else :size="22" />
    </div>

    <div class="message-column">
      <article
        class="message-bubble"
        :class="{
          pending: message.pending,
          error: message.error,
        }"
      >
        <div
          v-if="message.content"
          class="message-content"
        >
          {{ message.content }}
        </div>

        <div
          v-else-if="message.pending"
          class="typing-indicator"
        >
          <span />
          <span />
          <span />
        </div>

        <section
          v-if="
            message.role === 'assistant' &&
            message.sources.length > 0
          "
          class="message-sources"
        >
          <h3>来源文档</h3>
          <div class="source-grid">
            <SourceCard
              v-for="source in message.sources"
              :key="
                `${message.id}-${source.source_number}`
              "
              :source="source"
              :documents="documents"
            />
          </div>
        </section>
      </article>

      <footer
        v-if="message.role === 'assistant'"
        class="message-meta"
      >
        <span v-if="message.elapsedMs !== undefined">
          <Clock3 :size="15" />
          耗时 {{ elapsed(message.elapsedMs) }}
        </span>

        <button
          v-if="message.content"
          type="button"
          class="icon-button"
          :title="copied ? '已复制' : '复制回答'"
          @click="copyAnswer"
        >
          <Copy :size="15" />
        </button>
      </footer>
    </div>
  </div>
</template>
