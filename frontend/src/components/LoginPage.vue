<script setup lang="ts">
import { BookOpen, Eye, EyeOff, LoaderCircle, LockKeyhole, Mail } from "lucide-vue-next";
import { computed, ref } from "vue";

const props = defineProps<{
  busy: boolean;
  error?: string;
}>();

const emit = defineEmits<{
  login: [email: string, password: string];
  register: [email: string, password: string];
}>();

const mode = ref<"login" | "register">("login");
const email = ref("");
const password = ref("");
const showPassword = ref(false);

const title = computed(() => (mode.value === "login" ? "欢迎登录" : "创建账号"));
const subtitle = computed(() =>
  mode.value === "login" ? "登录并进入您的企业知识空间" : "注册后将自动登录 Knowledge Assistant",
);

function submit(): void {
  if (!email.value.trim() || !password.value || props.busy) return;
  if (mode.value === "login") {
    emit("login", email.value.trim(), password.value);
  } else {
    emit("register", email.value.trim(), password.value);
  }
}
</script>

<template>
  <main class="auth-page">
    <section class="auth-hero">
      <div class="auth-brand">
        <span class="brand-icon"><BookOpen :size="30" /></span>
        <div>
          <strong>知识库助手</strong>
          <span>Knowledge Assistant</span>
        </div>
      </div>

      <div class="auth-copy">
        <p class="eyebrow">Enterprise RAG Workspace</p>
        <h1>让企业私有知识<br />真正问起来</h1>
        <p>上传内部文档，自动完成解析、切片、向量化与检索，并通过可追溯来源回答问题。</p>
        <div class="auth-feature-grid">
          <article><strong>私有知识库</strong><span>按用户隔离知识空间</span></article>
          <article><strong>异步处理</strong><span>实时查看文档加工状态</span></article>
          <article><strong>流式问答</strong><span>SSE 实时返回答案</span></article>
          <article><strong>来源引用</strong><span>文件 / 章节 / 页码可追踪</span></article>
        </div>
      </div>
    </section>

    <section class="auth-panel-wrap">
      <form class="auth-card" @submit.prevent="submit">
        <div class="auth-card-heading">
          <span class="brand-icon small"><BookOpen :size="22" /></span>
          <h2>{{ title }}</h2>
          <p>{{ subtitle }}</p>
        </div>

        <label class="form-field">
          <span>邮箱</span>
          <div class="input-with-icon">
            <Mail :size="17" />
            <input v-model="email" type="email" autocomplete="email" placeholder="name@example.com" />
          </div>
        </label>

        <label class="form-field">
          <span>密码</span>
          <div class="input-with-icon">
            <LockKeyhole :size="17" />
            <input
              v-model="password"
              :type="showPassword ? 'text' : 'password'"
              :autocomplete="mode === 'login' ? 'current-password' : 'new-password'"
              :placeholder="mode === 'register' ? '至少 8 位密码' : '请输入密码'"
            />
            <button type="button" class="icon-button ghost" @click="showPassword = !showPassword">
              <EyeOff v-if="showPassword" :size="17" />
              <Eye v-else :size="17" />
            </button>
          </div>
        </label>

        <p v-if="error" class="form-error">{{ error }}</p>

        <button type="submit" class="primary-button auth-submit" :disabled="busy">
          <LoaderCircle v-if="busy" :size="18" class="spinning" />
          {{ mode === "login" ? "登录" : "注册并登录" }}
        </button>

        <div class="auth-switch">
          <span>{{ mode === "login" ? "还没有账号？" : "已有账号？" }}</span>
          <button type="button" @click="mode = mode === 'login' ? 'register' : 'login'">
            {{ mode === "login" ? "立即注册" : "返回登录" }}
          </button>
        </div>
      </form>
    </section>
  </main>
</template>
