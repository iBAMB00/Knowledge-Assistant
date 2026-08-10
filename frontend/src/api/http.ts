import axios from "axios";

const ACCESS_TOKEN_KEY = "knowledge-assistant-access-token";

export const http = axios.create({
  baseURL: "",
  timeout: 120_000,
  headers: {
    Accept: "application/json",
  },
});

http.interceptors.request.use((config) => {
  const token = getAccessToken();

  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }

  return config;
});

http.interceptors.response.use(
  (response) => response,
  (error) => {
    if (axios.isAxiosError(error) && error.response?.status === 401) {
      clearAccessToken();
      window.dispatchEvent(new CustomEvent("knowledge-assistant:unauthorized"));
    }

    return Promise.reject(error);
  },
);

export function getAccessToken(): string | null {
  return localStorage.getItem(ACCESS_TOKEN_KEY);
}

export function setAccessToken(token: string): void {
  localStorage.setItem(ACCESS_TOKEN_KEY, token);
}

export function clearAccessToken(): void {
  localStorage.removeItem(ACCESS_TOKEN_KEY);
}

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
      return "无法连接后端，请确认 Knowledge Assistant API 已启动。";
    }
  }

  if (error instanceof Error && error.message) {
    return error.message;
  }

  return "请求失败，请稍后重试。";
}
