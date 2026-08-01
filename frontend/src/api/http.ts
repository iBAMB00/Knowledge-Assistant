import axios from "axios";

export const http = axios.create({
  baseURL: "",
  timeout: 120_000,
  headers: {
    Accept: "application/json",
  },
});

export function getApiErrorMessage(error: unknown): string {
  if (axios.isAxiosError(error)) {
    const detail = error.response?.data?.detail;

    if (typeof detail === "string" && detail.trim()) {
      return detail;
    }

    if (Array.isArray(detail)) {
      const message = detail
        .map((item) => item?.msg)
        .find((item) => typeof item === "string");

      if (message) {
        return message;
      }
    }

    if (error.code === "ECONNABORTED") {
      return "请求超时，请稍后重试。";
    }

    if (!error.response) {
      return "无法连接后端，请确认 FastAPI 服务已启动。";
    }
  }

  if (error instanceof Error && error.message) {
    return error.message;
  }

  return "请求失败，请稍后重试。";
}
