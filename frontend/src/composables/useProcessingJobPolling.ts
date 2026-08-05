import {
  computed,
  onBeforeUnmount,
  onMounted,
  ref,
} from "vue";
import { getProcessingJob } from "@/api/knowledge";
import type {
  ActiveProcessingJob,
  DocumentRecord,
  ProcessingJobRecord,
  ProcessingJobSnapshot,
} from "@/types/knowledge";

const DEFAULT_POLL_INTERVAL_MS = 1_800;
const MAX_POLL_INTERVAL_MS = 10_000;

interface ProcessingJobPollingOptions {
  onTerminalJobs?: (
    jobs: ProcessingJobSnapshot[],
  ) => void | Promise<void>;
  onPollError?: (error: unknown) => void;
}

export function useProcessingJobPolling(
  options: ProcessingJobPollingOptions = {},
) {
  const jobsByDocumentId = ref<
    Record<number, ProcessingJobSnapshot>
  >({});

  const activeJobCount = computed(() =>
    Object.values(jobsByDocumentId.value).filter(
      isActiveProcessingJob,
    ).length,
  );

  let timerId: number | undefined;
  let pollInFlight = false;
  let consecutiveFailures = 0;
  let disposed = false;

  function syncDocuments(
    documents: DocumentRecord[],
  ): void {
    const documentIds = new Set(
      documents.map((document) => document.id),
    );

    for (const documentId of Object.keys(
      jobsByDocumentId.value,
    ).map(Number)) {
      if (!documentIds.has(documentId)) {
        delete jobsByDocumentId.value[documentId];
      }
    }

    for (const document of documents) {
      if (!document.active_job) {
        continue;
      }

      const current =
        jobsByDocumentId.value[document.id];

      if (
        !current ||
        current.id !== document.active_job.id ||
        isActiveProcessingJob(current)
      ) {
        jobsByDocumentId.value[document.id] =
          toProcessingJobSnapshot(
            document.id,
            document.active_job,
          );
      }
    }

    scheduleNextPoll(0);
  }

  function trackJob(job: ProcessingJobRecord): void {
    jobsByDocumentId.value[job.document_id] = {
      ...job,
    };

    scheduleNextPoll(0);
  }

  function forgetDocument(documentId: number): void {
    delete jobsByDocumentId.value[documentId];
  }

  async function pollActiveJobs(): Promise<void> {
    clearPollTimer();

    if (
      disposed ||
      pollInFlight ||
      document.visibilityState === "hidden"
    ) {
      return;
    }

    const activeJobs = Object.values(
      jobsByDocumentId.value,
    ).filter(isActiveProcessingJob);

    if (activeJobs.length === 0) {
      return;
    }

    pollInFlight = true;

    try {
      const results = await Promise.allSettled(
        activeJobs.map((job) =>
          getProcessingJob(job.id),
        ),
      );

      const terminalJobs: ProcessingJobSnapshot[] = [];
      let failedRequestCount = 0;

      for (const result of results) {
        if (result.status === "rejected") {
          failedRequestCount += 1;
          continue;
        }

        const job = result.value;
        jobsByDocumentId.value[job.document_id] = {
          ...job,
        };

        if (!isActiveProcessingJob(job)) {
          terminalJobs.push(job);
        }
      }

      if (failedRequestCount > 0) {
        consecutiveFailures += 1;

        if (consecutiveFailures === 1) {
          options.onPollError?.(
            new Error("processing job polling failed"),
          );
        }
      } else {
        consecutiveFailures = 0;
      }

      if (terminalJobs.length > 0) {
        await options.onTerminalJobs?.(terminalJobs);
      }
    } finally {
      pollInFlight = false;

      const delay = Math.min(
        DEFAULT_POLL_INTERVAL_MS *
          2 ** consecutiveFailures,
        MAX_POLL_INTERVAL_MS,
      );

      scheduleNextPoll(delay);
    }
  }

  function scheduleNextPoll(
    delay = DEFAULT_POLL_INTERVAL_MS,
  ): void {
    clearPollTimer();

    if (
      disposed ||
      document.visibilityState === "hidden" ||
      activeJobCount.value === 0
    ) {
      return;
    }

    timerId = window.setTimeout(() => {
      void pollActiveJobs();
    }, delay);
  }

  function clearPollTimer(): void {
    if (timerId === undefined) {
      return;
    }

    window.clearTimeout(timerId);
    timerId = undefined;
  }

  function handleVisibilityChange(): void {
    if (document.visibilityState === "hidden") {
      clearPollTimer();
      return;
    }

    scheduleNextPoll(0);
  }

  onMounted(() => {
    document.addEventListener(
      "visibilitychange",
      handleVisibilityChange,
    );
  });

  onBeforeUnmount(() => {
    disposed = true;
    clearPollTimer();
    document.removeEventListener(
      "visibilitychange",
      handleVisibilityChange,
    );
  });

  return {
    jobsByDocumentId,
    activeJobCount,
    syncDocuments,
    trackJob,
    forgetDocument,
  };
}

export function isActiveProcessingJob(
  job: Pick<ProcessingJobSnapshot, "status">,
): boolean {
  return (
    job.status === "pending" ||
    job.status === "running"
  );
}

function toProcessingJobSnapshot(
  documentId: number,
  job: ActiveProcessingJob,
): ProcessingJobSnapshot {
  return {
    ...job,
    document_id: documentId,
  };
}
