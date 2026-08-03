# Knowledge Assistant v0.9.0 Release Notes

发布日期：2026-08-04  
版本定位：可演示、可验证、可用于面试展示的单机后端 RAG MVP

## 版本概览

`v0.9.0` 完成了 Knowledge Assistant 的第一条完整企业知识库问答链路：

```text
文档上传
→ 文档解析 / OCR
→ Chunk
→ Embedding
→ Dense Retrieval
→ ContextBuilder
→ Knowledge Chat
→ 来源追溯
→ 普通响应或 SSE
```

同时加入 ProcessingJob 任务管理、Alembic 数据库迁移、测试数据库隔离以及 SQLite 外键和删除级联保护。

## 主要能力

### 文档加工

- 文档上传、列表、详情和删除；
- TXT 文档解析；
- PDF 文本层提取；
- PDF 异常文本检测；
- 扫描件或异常 PDF 的 OCR 降级；
- 中文和英文 OCR；
- Recursive Character Chunk；
- Embedding 生成与 SQL 持久化。

### 检索与问答

- Dense Retrieval；
- Candidate-K 扩召回；
- 相似度阈值过滤；
- 检索结果去重；
- 多文档结果平衡；
- ContextBuilder；
- 无有效上下文时直接拒答；
- 非流式 Knowledge Chat；
- SSE 流式 Knowledge Chat；
- 回答来源和真实文件名追溯。

### ProcessingJob

支持：

```text
document_processing
embedding
full_pipeline
```

具备：

- 任务创建、详情、最新任务和历史查询；
- `pending / running / succeeded / failed` 状态；
- 任务进度和失败原因；
- 同一文档只允许一个活动任务；
- 失败后允许创建新任务重试；
- 完整处理流水线。

### 工程能力

- FastAPI Router / Service / Repository 分层；
- Service 控制事务，Repository 只执行数据读写和 `flush`；
- SQLAlchemy + SQLite；
- Alembic 管理数据库结构；
- 独立 pytest 测试数据库；
- SQLite 每个连接开启 `PRAGMA foreign_keys=ON`；
- 删除文档时级联清理 Content、Chunk、Embedding 和 ProcessingJob。

## 验收结果

### 自动化测试

```text
157 passed
1 skipped
1 warning
```

唯一 warning 来自 Starlette `TestClient` 与 `httpx` 的兼容性弃用提示，不影响本版本业务验收。

### 独立空数据库 E2E

以下链路已通过：

```text
创建独立空 SQLite 数据库
→ alembic upgrade head
→ 启动 API
→ 上传固定验收文档
→ 创建 full_pipeline Job
→ Job succeeded / progress 100
→ DocumentContent 正常
→ Chunk 正常
→ Embedding 正常
→ Knowledge Chat 正常
→ 来源文件名正确
→ 无答案问题拒答
→ SSE metadata → message → done
→ 删除 Document
→ 所有关联数据级联清理
```

数据库最终检查：

```text
PRAGMA foreign_keys = 1
PRAGMA foreign_key_check = []

documents = 0
document_contents = 0
document_chunks = 0
chunk_embeddings = 0
processing_jobs = 0
```

## 配置与运行

数据库初始化：

```powershell
alembic upgrade head
```

启动服务：

```powershell
uvicorn app.main:app --reload
```

Swagger：

```text
http://127.0.0.1:8000/docs
```

完整环境变量和 OCR 安装说明见根目录 `README.md`。

## 已知限制

本版本仍是单机后端 MVP，尚未完成：

- Qdrant 专用向量数据库；
- BM25 / Hybrid Retrieval / RRF；
- Cross-Encoder Reranker；
- Parent-Child Chunk；
- 独立 Worker、任务超时和恢复；
- PostgreSQL、Redis 和 Celery；
- MinIO；
- Docker Compose；
- 完整前端；
- 多租户、认证和复杂权限；
- 生产级日志、监控、限流和安全防护。

本版本不应描述为生产级、高并发或可直接商业化交付的企业系统。

## 下一版本

`v0.10.0` 的准确主线是 Qdrant：

```text
Qdrant 配置和客户端生命周期
QdrantVectorStore
Collection 初始化
Dense Upsert
Dense Search
document_id Filter
按 document_id Delete
重复 Upsert 幂等
单文档和全量索引重建
DatabaseVectorStore 保留为 Baseline
Docker 启动 Qdrant
```
