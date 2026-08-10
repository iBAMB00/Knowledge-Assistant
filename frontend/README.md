# Knowledge Assistant Frontend v1.0

Vue 3 + Vite + TypeScript frontend for the Knowledge Assistant v1.0 release acceptance flow.

## Features

- Register / Login / JWT session restore
- Knowledge Base CRUD
- Knowledge Base document upload and list
- Automatic `full_pipeline` ProcessingJob creation after upload
- ProcessingJob polling and status page
- Knowledge Chat and SSE streaming
- Source citation with filename / chunk / section / page metadata
- Knowledge Base settings
- Account information and light/dark theme
- Responsive desktop/mobile layout

## Local development

Backend should be available at `http://127.0.0.1:8000`.

```bash
npm install
npm run dev
```

Open `http://127.0.0.1:5173`.

The Vite development proxy forwards `/auth`, `/knowledge-bases`, `/documents`, `/processing-jobs`, `/knowledge`, and `/health` to the FastAPI backend.

## Build

```bash
npm run build
```
