<script setup lang="ts">
import {
  Database,
  FileUp,
  LoaderCircle,
  Send,
  Square,
} from "lucide-vue-next";
import { ref } from "vue";
import type {
  DocumentRecord,
} from "@/types/knowledge";

const props = defineProps<{
  documents: DocumentRecord[];
  selectedDocumentId?: number;
  submitting: boolean;
  streamingEnabled: boolean;
  uploadBusy: boolean;
  uploadProgress: number | null;
}>();

const emit = defineEmits<{
  send: [question: string];
  stop: [];
  upload: [file: File];
  "update:selectedDocumentId": [value?: number];
  "update:streamingEnabled": [value: boolean];
}>();

const question = ref("");
const fileInput =
  ref<HTMLInputElement | null>(null);

function submit(): void {
  const normalized = question.value.trim();

  if (!normalized || props.submitting) {
    return;
  }

  emit("send", normalized);
  question.value = "";
}

function onKeydown(event: KeyboardEvent): void {
  if (event.key !== "Enter" || event.shiftKey) {
    return;
  }

  event.preventDefault();
  submit();
}

function chooseFile(): void {
  if (!props.uploadBusy) {
    fileInput.value?.click();
  }
}

function onFileSelected(event: Event): void {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];

  if (file && !props.uploadBusy) {
    emit("upload", file);
  }

  input.value = "";
}

function onDocumentChange(event: Event): void {
  const value = (
    event.target as HTMLSelectElement
  ).value;

  emit(
    "update:selectedDocumentId",
    value ? Number(value) : undefined,
  );
}

function onStreamingChange(event: Event): void {
  emit(
    "update:streamingEnabled",
    (event.target as HTMLInputElement).checked,
  );
}
</script>

<template>
  <section class="composer">
    <textarea
      v-model="question"
      rows="3"
      maxlength="2000"
      :disabled="submitting"
      placeholder="输入您的问题，按 Enter 发送，Shift + Enter 换行"
      @keydown="onKeydown"
    />

    <div class="composer-footer">
      <div class="composer-actions">
        <label class="document-select">
          <Database :size="17" />
          <select
            :value="selectedDocumentId ?? ''"
            @change="onDocumentChange"
          >
            <option value="">全部知识库</option>
            <option
              v-for="document in documents"
              :key="document.id"
              :value="document.id"
            >
              {{ document.filename }}
            </option>
          </select>
          <i />
        </label>

        <button
          type="button"
          class="secondary-button"
          :disabled="uploadBusy"
          @click="chooseFile"
        >
          <LoaderCircle
            v-if="uploadBusy"
            :size="17"
            class="spinning"
          />
          <FileUp v-else :size="17" />
          {{
            uploadBusy
              ? `上传 ${uploadProgress ?? 0}%`
              : "上传文件"
          }}
        </button>

        <label class="stream-switch">
          <input
            type="checkbox"
            :checked="streamingEnabled"
            @change="onStreamingChange"
          />
          <span>流式回答</span>
        </label>

        <input
          ref="fileInput"
          class="visually-hidden"
          type="file"
          accept=".txt,.md,.pdf"
          :disabled="uploadBusy"
          @change="onFileSelected"
        />
      </div>

      <div class="composer-submit">
        <span>{{ question.length }} / 2000</span>

        <button
          v-if="submitting"
          type="button"
          class="stop-button"
          @click="emit('stop')"
        >
          <Square :size="15" />
          停止
        </button>

        <button
          v-else
          type="button"
          class="primary-button"
          :disabled="!question.trim()"
          @click="submit"
        >
          <Send :size="18" />
          发送
        </button>
      </div>
    </div>
  </section>
</template>
