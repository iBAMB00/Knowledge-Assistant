import { fileURLToPath, URL } from "node:url";
import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: {
      "/auth": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/knowledge-bases": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/documents": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/knowledge": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/agent": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/conversations": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/processing-jobs": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/health": { target: "http://127.0.0.1:8000", changeOrigin: true },
    },
  },
});
