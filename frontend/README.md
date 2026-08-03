# Knowledge Assistant Frontend

方案 A 的 Vue 3 + TypeScript + Vite 演示前端。

## 已实现

- 知识库普通问答；
- POST SSE 流式问答；
- 来源卡片；
- 指定文档问答；
- 文档列表；
- 文档上传；
- 文档解析与切片；
- 文档向量化；
- 文档删除；
- 知识库统计；
- 深色模式；
- 响应式页面。

## 替换

压缩包根目录包含 `frontend/`。

建议先把项目中的原 `frontend` 重命名为
`frontend_backup`，再把压缩包里的 `frontend`
复制到 Knowledge-Assistant 项目根目录。

## 启动

后端：

```powershell
uvicorn app.main:app --reload
```

前端：

```powershell
cd frontend
npm install
npm run dev
```

浏览器：

```text
http://127.0.0.1:5173
```

## 验证

```powershell
npm run type-check
npm run build
```

## 接口

- `GET /documents/`
- `POST /documents/`
- `POST /documents/{id}/process`
- `POST /documents/{id}/embeddings`
- `DELETE /documents/{id}`
- `GET /documents/{id}/chunk-summary`
- `POST /knowledge/chat`
- `POST /knowledge/chat/stream`

流式接口使用 `fetch + ReadableStream` 解析 POST SSE，
不是原生 `EventSource`，因为 `EventSource` 只支持 GET。

Vite 已代理 `/documents` 和 `/knowledge` 到
`http://127.0.0.1:8000`，开发阶段无需新增 CORS 配置。
