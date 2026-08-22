<script setup lang="ts">
import {
  Bot,
  CheckCircle2,
  CircleDot,
  PlugZap,
  Wrench,
  XCircle,
} from "lucide-vue-next";
import type { AgentActivityRecord } from "@/types/knowledge";

defineProps<{ activities: AgentActivityRecord[] }>();

function displayToolName(toolName: string): string {
  if (!toolName.startsWith("mcp__")) return toolName;
  const [, server, ...toolParts] = toolName.split("__");
  return [server, toolParts.join("__")].filter(Boolean).join(" / ") || toolName;
}
</script>

<template>
  <section v-if="activities.length" class="agent-activity-panel">
    <div class="agent-activity-heading">
      <Bot :size="14" />
      <strong>Agent 运行轨迹</strong>
      <span>仅展示安全生命周期事件</span>
    </div>

    <div class="agent-activity-list">
      <div v-for="activity in activities" :key="activity.id" class="agent-activity-item">
        <template v-if="activity.kind === 'status'">
          <span class="activity-icon activity-model"><CircleDot :size="13" /></span>
          <div class="activity-copy">
            <strong>第 {{ activity.turn }} 轮模型决策</strong>
            <span>正在判断是否需要调用工具</span>
          </div>
          <span class="activity-state">分析</span>
        </template>

        <template v-else>
          <span class="activity-icon" :class="`activity-${activity.status}`">
            <PlugZap v-if="activity.provider === 'mcp'" :size="13" />
            <Wrench v-else :size="13" />
          </span>
          <div class="activity-copy">
            <div class="activity-tool-line">
              <strong>{{ displayToolName(activity.toolName) }}</strong>
              <span class="tool-provider-badge" :class="`provider-${activity.provider}`">
                {{ activity.provider === 'mcp' ? 'MCP' : '本地工具' }}
              </span>
            </div>
            <span v-if="activity.status === 'running'">正在执行受控工具调用</span>
            <span v-else-if="activity.status === 'succeeded'">工具执行完成</span>
            <span v-else>工具执行失败<span v-if="activity.errorCode"> · {{ activity.errorCode }}</span></span>
            <code>{{ activity.toolName }}</code>
          </div>
          <span class="activity-state" :class="`state-${activity.status}`">
            <CheckCircle2 v-if="activity.status === 'succeeded'" :size="12" />
            <XCircle v-else-if="activity.status === 'failed'" :size="12" />
            <CircleDot v-else :size="12" />
            {{ activity.status === 'succeeded' ? '完成' : activity.status === 'failed' ? '失败' : '执行中' }}
          </span>
        </template>
      </div>
    </div>
  </section>
</template>
