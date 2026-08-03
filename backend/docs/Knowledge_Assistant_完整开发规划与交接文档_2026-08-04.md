# Knowledge Assistant 项目完整开发规划与交接文档

> 项目名称：Knowledge Assistant  
> 文档类型：项目事实基线 + 架构交接 + 开发进度 + 面试优先路线 + 版本规划 + 技术债  
> 最近更新：2026-08-04  
> 当前版本：`v0.9.0`  
> 当前产品阶段：后端 RAG MVP，技术验收已完成，待创建 Git Tag / Release  
> 当前主线：以面试价值为优先，升级为可评估、可部署、可解释的企业级 RAG 系统  
> 下一准确开发块：Qdrant 向量存储选型与接入设计  
> 文档用途：新对话无缝恢复项目上下文的统一事实基线

---

## 一、文档使用规则

本文件替代 2026-07-30 版本的交接文档，合并了：

1. 原有项目定位、开发规范、技术环境和架构原则；
2. 文档解析、OCR、Chunk、Embedding 和检索的历史进展；
3. 已完成的单轮非流式与流式 RAG 闭环；
4. 已完成的 ProcessingJob 异步任务体系；
5. 已完成的来源文件名全链路追溯；
6. `score_threshold` 的内外部配置边界；
7. 当前面试可用度和真实版本定位；
8. Qdrant、企业级评估、混合检索、Rerank、Parent-Child、Worker 和 Docker 的新规划；
9. 大幅简化后的测试策略；
10. 下一开发块需要检查的真实代码范围。

后续若旧文档、旧对话或早期计划与本文件冲突，以本文件中日期更新更晚、范围更明确的内容为准。

项目命名统一规则：

```text
正式产品名称：Knowledge Assistant
中文定位：企业私有知识库 RAG 助手
历史名称：Secure Assistant，仅允许出现在历史提交、旧文档说明或技术环境名中
现有技术标识：secure_assistant.db、Secure-Assistant Conda 环境等暂不强制重命名
```

状态标记统一使用：

```text
DONE        已完成并经过基本验证
VERIFY      代码已完成，等待最终验收结果
IN PROGRESS 当前正在处理
NEXT        下一准确开发块
PLANNED     已排期但尚未开始
DEFERRED    明确延后
BLOCKED     受外部条件阻塞
```

---

## 二、协作角色与指导方式

新对话中的助手应同时承担以下角色：

```text
资深软件架构师
资深 Python / FastAPI 后端开发者
企业级 RAG 应用架构师
AI 应用岗位面试项目导师
项目进度和技术债维护者
```

指导目标不是单纯补代码，而是帮助用户真正理解：

- 请求链路；
- 模块职责；
- 事务边界；
- 数据一致性；
- 检索质量；
- 系统演进；
- 技术选型；
- 面试表达。

### 2.1 开发前必须执行的流程

```text
检查最新项目目录
→ 确认相关模块是否已经存在
→ 阅读当前完整代码
→ 恢复真实调用链
→ 分析职责、架构影响和取舍
→ 确定最小修改范围
→ 再输出代码
```

具体要求：

1. 不默认项目中缺少某个模块；
2. 不根据记忆重建已有文件；
3. 已知文件存在但没有最新完整代码时，先请用户提交；
4. 新文件只有在确认不存在且职责必要时才能创建；
5. 优先复用现有抽象，不另起平行架构；
6. 修改 Router 前检查路由注册和路径冲突；
7. 修改 Service 前检查上游 API、下游 Repository 和事务边界；
8. 修改 Schema 前检查同步接口、流式接口、调试接口和测试的公共契约；
9. 一轮开发所需代码尽量集中一次给出；
10. 小改动只给相关片段，不重复无关完整文件。

### 2.2 每个开发块开始前固定输出

```text
本轮目标
当前真实请求链路
模块职责
架构问题和风险
设计取舍
事务边界
受影响文件
不在本轮处理的事项
测试计划
验收标准
```

### 2.3 每个开发块结束时固定输出

```text
完成内容
修改文件
关键架构决策
测试命令
测试结果
验收标准
已解决技术债
新增技术债及目标版本
下一准确步骤
推荐中文 Git 提交信息
```

### 2.4 代码输出风格

- 新增类和方法应有基本注释或文档字符串；
- 不机械地把每个参数和表达式拆成一行；
- 能清晰写成一至两行时保持紧凑；
- 复杂条件可按语义分行；
- 代码必须贴合现有项目结构；
- 不为“看起来企业级”而增加无价值的类和目录。

---

## 三、API、事务和工程规范

### 3.1 API 响应规范

```text
成功：
只返回必要业务字段。

失败：
通过 HTTP 状态码和 detail 表达。

禁止：
统一使用 code/message/data 包装。
```

至少区分：

```text
400：业务参数内容不合法
404：资源不存在
409：资源状态冲突或并发冲突
422：Pydantic 请求校验失败
500：未知服务端错误
503：必要外部服务不可用
```

### 3.2 数据库事务规范

```text
Repository：
只执行查询、add、update、delete、flush。
不自行 commit。
不负责业务状态机。
不抛 HTTPException。

Service：
负责编排业务事务。
负责 commit / rollback。
负责状态转换和跨 Repository 协作。
```

### 3.3 基础设施抽象原则

```text
业务 Service
不直接依赖本地磁盘路径
不直接依赖具体向量数据库
不直接依赖具体 Embedding Provider
不直接依赖具体任务队列
```

关键抽象：

```text
StorageService
EmbeddingProvider
VectorStore
ProcessingJobDispatcher（后续增加）
Reranker（后续增加）
QueryRewriter（后续增加）
```

---

## 四、项目定位与职业目标

### 4.1 产品定位

> 帮助企业快速搭建私有知识问答助手，让企业内部文档、产品手册、流程、规范、API 文档和技术资料真正可以被检索和问答。

重点能力：

```text
企业私有数据加工
企业文档检索
企业知识问答
新员工快速熟悉资料
内部流程和规范查询
API / 技术文档查询
答案来源追溯
```

不以通用写作、通用闲聊和纯聊天机器人为核心能力。

### 4.2 用户的职业目标

用户原有 Java 后端经验，正在转向：

```text
Java + AI 应用开发
Python AI 应用开发
RAG 应用开发
大模型应用工程师
智能体应用开发
AI 后端工程师
```

因此项目开发优先级从“补齐所有产品功能”调整为：

> 优先完成对面试有解释价值、可演示价值和工程价值的能力。

项目需要证明用户掌握：

- RAG 全链路，而不只是调用 LangChain；
- 文档处理和结构化切片；
- Embedding 与向量库建库；
- Dense、Sparse、Hybrid、Rerank；
- 检索评估；
- 异步任务；
- Docker 部署；
- 数据一致性和失败恢复；
- 企业级架构边界。

---

## 五、当前版本与阶段判断

### 5.1 当前版本定位

当前不建议正式标记为生产意义上的 `V1.0`。

建议定位：

```text
当前版本：v0.9.0
发布状态：技术验收已完成，待创建 Git Tag / Release
产品阶段：Backend RAG MVP
项目性质：可演示、可验证、可用于面试的单机后端版本
```

### 5.2 完成度估算

```text
后端 RAG MVP：约 90%
面试演示版本：约 80%～85%
可部署单租户 V1：约 55%～65%
企业生产级产品：尚未完成
```

以上为项目管理估算，不是代码覆盖率。

### 5.3 当前能否用于面试

结论：

> 可以，并且应当开始用于投递和面试。

当前项目已经明显超过普通“上传 PDF + 调模型”的 Demo，具备：

- 清晰的后端分层；
- 自研文档加工链路；
- OCR 降级；
- Chunk、Embedding、Retrieval、Context、RAG；
- 同步和 SSE；
- 真实来源；
- 异步 ProcessingJob；
- 并发约束和失败重试；
- 检索评估框架；
- 较完整测试。

但面试时应准确表达为：

> 我从零实现了一套企业知识库 RAG 后端 MVP，完整覆盖文档上传、解析、OCR、切片、向量化、检索优化、受控问答、来源追溯、SSE 和异步处理任务。目前本地单机版本可以完整演示，正在继续升级 Qdrant、混合检索、Reranker、企业级评估、独立 Worker 和 Docker 部署。

不要声称：

```text
已经是生产级企业系统
已经支持大规模高并发
已经完成多租户和完整权限
已经可以直接商业化交付
已经完成企业级 Agent 平台
```

---

## 六、当前技术环境

```text
操作系统：Windows
Conda 环境：Secure-Assistant
Python：3.11.15
Web 框架：FastAPI
ORM：SQLAlchemy
关系数据库：SQLite
数据库迁移：Alembic
测试：pytest 9.1.1
运行：uvicorn app.main:app --reload
```

主要依赖：

```text
fastapi
uvicorn
sqlalchemy
alembic
pydantic
openai
pytest
python-multipart
PyMuPDF
```

OCR：

```text
Tesseract OCR
eng.traineddata
chi_sim.traineddata
```

`TESSDATA_PREFIX` 已在 Conda 环境中配置。README 后续仍应补充完整安装步骤。

当前外部模型资源包括 DeepSeek、火山方舟或兼容 OpenAI SDK 的模型服务。ChatGPT Plus 不等同于 OpenAI API 额度。

---

## 七、当前完整系统链路

### 7.1 文档加工与问答主链路

```text
上传企业文档
  ↓
StorageService 保存原始文件
  ↓
Document 保存元数据
  ↓
创建 ProcessingJob
  ↓
DocumentProcessingService
  ↓
ParserService
  ├─ TXT Parser
  ├─ PDF 文本层解析
  └─ PDF 乱码 / 扫描件 OCR 降级
  ↓
DocumentContent
  ↓
ChunkService / ChunkStrategy
  ↓
DocumentChunk
  ↓
EmbeddingService
  ↓
ChunkEmbedding
  ↓
VectorStore
  ↓
RetrievalService
  ↓
ContextBuilder
  ↓
KnowledgeChatService
  ↓
LLMService
  ↓
普通响应或 SSE
  ↓
答案 + 真实来源
```

### 7.2 当前数据关系

```text
Document
  ↓
DocumentContent
  ↓
DocumentChunk
  ↓
ChunkEmbedding

Document
  ↓
ProcessingJob
```

### 7.3 知识库问答链路

```text
POST /knowledge/chat
或
POST /knowledge/chat/stream
  ↓
KnowledgeChat API
  ↓
KnowledgeChatService.prepare
  ↓
RetrievalService.retrieve
  ↓
EmbeddingProvider.embed_query
  ↓
VectorStore.search
  ↓
阈值过滤、去重、多文档平衡
  ↓
ContextBuilder
  ↓
构造 Prompt 和公开 Sources
  ↓
无上下文：直接拒答，不调用 LLM
有上下文：LLM 普通或流式生成
```

---

## 八、当前确认的核心模块

以下为已确认存在或已确认职责的核心模块。具体目录以用户最新 `tree /F` 为准。

### 8.1 API

```text
app/api/chat.py
app/api/knowledge.py
app/api/knowledge_chat.py
app/api/retrieval.py
app/api/processing_job.py
```

### 8.2 核心 Service

```text
ChatService
LLMService
DocumentService
DocumentProcessingService
ParserService
ChunkService
EmbeddingService
RetrievalService
KnowledgeChatService
ProcessingJobService
ProcessingJobExecutor
ProcessingJobRunner
```

### 8.3 基础设施与算法模块

```text
app/services/storage_service.py
app/services/chunking/
app/services/embedding/
app/services/vector_store/base.py
app/services/vector_store/database.py
app/services/rag/context_builder.py
app/services/evaluation/
```

### 8.4 Repository

```text
DocumentRepository
DocumentContentRepository
DocumentChunkRepository
ChunkEmbeddingRepository
ProcessingJobRepository
```

### 8.5 数据模型

```text
Document
DocumentContent
DocumentChunk
ChunkEmbedding
ProcessingJob
```

### 8.6 关键测试文件

```text
tests/conftest.py
tests/test_document_service.py
tests/test_document_processing.py
tests/test_parser_service.py
tests/test_chunk.py
tests/test_embedding.py
tests/test_vector_store.py
tests/test_retrieval.py
tests/test_retrieval_api.py
tests/test_retrieval_case_loader.py
tests/test_retrieval_evaluator.py
tests/test_context_builder.py
tests/test_knowledge_chat.py
tests/test_knowledge_chat_api.py
tests/test_knowledge_chat_stream.py
tests/test_processing_job.py
```

规则：所有 Chunk 相关测试继续统一放在 `tests/test_chunk.py`。

---

## 九、已完成开发进度

## 9.1 基础工程

状态：`DONE`

- FastAPI 应用初始化；
- Router 注册；
- Pydantic Schema；
- SQLAlchemy；
- SQLite；
- Alembic；
- `.env` / Settings；
- 独立 pytest 测试数据库；
- 标准 HTTP 语义；
- 普通 LLM 调用；
- SSE 流式输出；
- Router、Service、Repository 基础分层；
- Service 控制事务；
- Repository 不 commit。

## 9.2 文档生命周期

状态：`DONE（核心）`

已完成：

- 文档上传；
- 文档列表；
- 文档详情；
- 文档删除；
- 本地文件存储；
- 文档元数据保存；
- 文档状态流转；
- 文档内容查询；
- Chunk 查询或摘要；
- 删除关联内容的基础一致性处理。

当前决定：

> 文档生命周期只完成核心即可，不继续投入复杂版本、软删除和内容治理，除非阻塞新主线。

后续可快速增加：

- MinIO 实现；
- `path` 向 `storage_key` 演进；
- 文件与数据库一致性补偿。

## 9.3 Parser 与 OCR

状态：`DONE（MVP）`

支持：

```text
TXT
PDF 文本层
PDF 字体映射异常检测
扫描 PDF OCR
中文 + 英文 OCR
空白页跳过
全文为空时失败
```

Parser 版本历史建议：

```text
PDF_PARSER_VERSION = 1.2.0
TXT_PARSER_VERSION = 1.1.0
```

当前不继续建设复杂 OCR CleaningPipeline。

已记录技术债：

- 页眉页脚清理；
- 多栏布局恢复；
- 表格结构恢复；
- 坐标级文本重建；
- 代码缩进保护；
- 文本质量评分。

## 9.4 Chunk V1

状态：`DONE`

当前策略：

```text
recursive_character
```

已完成：

- 自然分隔符优先；
- Overlap 从自然边界开始；
- offset 与文本一致；
- 尾部碎片处理；
- 防止死循环；
- Chunk 状态和持久化；
- Chunk 核心测试。

已有 24 条检索用例回归曾得到满分，但只证明没有回归，不能证明策略已经企业级最优。

## 9.5 Embedding 与当前向量持久化

状态：`DONE（MVP）`

已完成：

- EmbeddingProvider 抽象；
- 多 Provider 工厂；
- 文档批量 Embedding；
- Query Embedding；
- ChunkEmbedding 持久化；
- 模型名、维度和向量合法性校验；
- save_or_update；
- Embedding 状态管理。

当前真实向量存储：

```text
SQLite / SQLAlchemy
+ JSON 保存向量
+ Python 计算余弦相似度
```

注意：

> 当前不是没有向量化，而是已经完成向量化和向量持久化，但还没有使用专用向量数据库建索引。

## 9.6 RetrievalService

状态：`DONE（MVP）`

已完成：

- Dense 向量检索；
- Top-K；
- Candidate-K；
- `score_threshold`；
- `document_id` 过滤；
- 重复内容过滤；
- 单文档数量限制；
- 结果不足回填；
- 多文档平衡；
- Baseline 模式；
- Optimized 模式；
- Debug Retrieval API；
- RetrievalEvaluator 框架；
- 非法模式和参数校验。

## 9.7 ContextBuilder

状态：`DONE`

已完成：

- 结果按分数排序；
- 空内容过滤；
- 重复内容过滤；
- 最大 Chunk 数；
- 最大字符数；
- 截断；
- 稳定来源编号；
- 公开来源摘要；
- 文件名进入上下文；
- 不调用数据库和 LLM。

## 9.8 Knowledge Chat

状态：`DONE`

非流式：

```text
POST /knowledge/chat
```

流式：

```text
POST /knowledge/chat/stream
```

已完成：

- 单轮 RAG 问答；
- Prompt 只允许基于知识库回答；
- 无可靠上下文直接拒答；
- 无召回时不调用 LLM；
- 普通回答；
- 流式回答；
- SSE metadata；
- SSE message；
- SSE done；
- SSE error；
- 客户端取消时关闭内容流；
- 模型空回答检测；
- 真实来源；
- 来源编号；
- 文件名；
- 摘要；
- 不暴露内部 Chunk、Score、路径和向量字段。

## 9.9 检索阈值配置边界

状态：`DONE`

业务决策：

```text
RETRIEVAL_SCORE_THRESHOLD=-1.0
→ 检索调试和评估使用

KNOWLEDGE_CHAT_SCORE_THRESHOLD=0.40
→ 实际知识库问答使用
```

普通 Knowledge Chat API 不允许外部传入 `score_threshold` 覆盖业务参数。

请求 Schema 使用禁止额外字段策略后，外部传入 `score_threshold` 应返回 422，而不是静默忽略。

`KnowledgeChatService` 内部可以保留可选 `score_threshold` 参数，供测试、评估或内部调用使用，但普通业务 Router 不透传该字段。

## 9.10 来源文件名全链路

状态：`DONE / VERIFY`

已完成链路：

```text
Document.filename
→ ChunkEmbeddingRepository JOIN
→ DatabaseVectorStore
→ VectorSearchResult.filename
→ ContextBuilder / ContextSource
→ KnowledgeChatService
→ KnowledgeChatSource.filename
→ 普通响应 / SSE metadata
→ 前端来源卡片
```

设计决策：

- 在候选查询阶段一次 JOIN 获取 Document；
- 不在 KnowledgeChatService 中逐条查询，避免 N+1；
- `filename` 是公开业务元数据；
- `chunk_id`、`score`、`path` 等仍不对外暴露；
- 同一文档多个 Chunk 仍可生成多个来源编号。

最近因为 `filename` 变为必填字段，多个测试 Fake 和构造函数出现校验失败。用户已经逐个补齐，但尚未提供最终全量 pytest 结果，因此本开发块状态标记为 `VERIFY`。

## 9.11 ProcessingJob 异步任务体系

状态：`DONE（单进程版本）`

任务类型：

```text
document_processing
embedding
full_pipeline
```

任务状态：

```text
pending
running
succeeded
failed
```

已完成：

- ProcessingJob 数据模型；
- Repository；
- Service；
- Executor；
- Runner；
- 创建处理任务；
- 按 Job ID 查询；
- 查询最新任务；
- 查询任务历史；
- 同一文档只允许一个活动任务；
- 数据库部分唯一索引；
- 重复创建返回 409；
- 失败后允许新任务重试；
- 失败保留最后进度；
- 完整流水线不会递归创建子任务；
- 最新任务重复路由已清理；
- 业务 Router 路由唯一性测试。

进度约定：

```text
document_processing：10 → 90 → 100
embedding：10 → 90 → 100
full_pipeline：10 → 60 → 90 → 100
```

当前调度载体仍主要依赖 FastAPI BackgroundTasks。


v0.9.0 删除一致性验收：

```text
SQLite 每个连接统一执行 PRAGMA foreign_keys=ON
Document 删除后级联清理 DocumentContent
DocumentContent 删除后级联清理 DocumentChunk
DocumentChunk 删除后级联清理 ChunkEmbedding
Document 删除后级联清理 ProcessingJob
PRAGMA foreign_key_check 返回空结果
```

## 9.12 测试体系

状态：`DONE，但准备精简`

已经具备：

- Service 测试；
- Repository / 数据层测试；
- Router API 测试；
- SSE 生命周期测试；
- Retrieval 算法测试；
- RetrievalEvaluator 测试；
- ProcessingJob 状态和并发测试；
- 测试数据库隔离；
- 参数和异常测试。

v0.9.0 最终全量回归结果：

```text
157 passed
1 skipped
1 warning
```

唯一 warning 来自 Starlette `TestClient` 与 `httpx` 的兼容性弃用提示，不影响当前业务验收，后续随依赖升级处理。

---

## 十、当前页面与演示状态

已看到的前端页面形态包括：

- 用户问题；
- 流式回答；
- `[来源 1]` 等引用；
- 来源卡片；
- 文件名；
- 摘要；
- 多来源；
- 耗时显示。

当前后端已经能够直接返回前端来源卡片所需字段：

```json
{
  "source_number": 1,
  "document_id": 2,
  "filename": "02_账号与权限管理规范.txt",
  "excerpt": "管理员可以在系统设置中重置用户密码……"
}
```

尚未确认完整前端目录和所有页面是否已经接入最新 API，因此不把前端完整度作为当前主线。

---

## 十一、当前架构的核心价值与限制

### 11.1 面试价值最高的已完成内容

1. 自研完整 RAG 链路；
2. Parser、OCR、Chunk、Embedding、VectorStore 分层；
3. Service 事务边界；
4. Baseline / Optimized 检索；
5. 多文档平衡和去重；
6. 无召回不调用 LLM；
7. 真实来源和文件名；
8. SSE 生命周期；
9. ProcessingJob 并发和失败重试；
10. 评估框架；
11. Alembic 和测试隔离。

### 11.2 当前主要限制

```text
向量仍保存在关系数据库 JSON 中
检索仍由 Python 暴力计算
没有专用 ANN 索引
没有 BM25 + Vector 混合检索
没有 Reranker
没有 Query Rewrite
评估集规模和难度不足
只支持 TXT / PDF
没有 Parent-Child Chunk
BackgroundTasks 不能跨进程可靠执行
SQLite 不适合 API + Worker 并发写入
没有 Docker Compose 完整部署
没有 PostgreSQL
没有 MinIO
没有完整生产可观测性
```

---

## 十二、新的开发总原则：面试价值优先

从 2026-08-04 起，开发主线调整为：

> 不再优先补全所有普通 CRUD 和外围产品功能，而是优先完成能体现企业 RAG 技术深度的关键节点。

优先级：

```text
P0：Qdrant 向量库
P0：企业级检索评估体系
P0：BM25 + Vector 混合检索
P0：Reranker

P1：Parent-Child 与结构感知切片
P1：Docker Compose
P1：PostgreSQL
P1：Celery + Redis Worker
P1：MinIO 快速接入

P2：Markdown / HTML / 表格 / 代码解析
P2：Query Rewrite / Multi-Query / HyDE
P2：网页采集

P3：权限、多租户、Agent、管理后台
```

---

## 十三、向量数据库规划

### 13.1 当前 DatabaseVectorStore 的保留价值

当前实现不要删除。它继续作为：

```text
精确检索基线
算法透明的参考实现
单元测试实现
Qdrant 结果对照基准
无外部依赖的本地模式
```

### 13.2 向量数据库选型比较

| 方案 | 主要优势 | 主要代价 | 本项目定位 |
|---|---|---|---|
| DatabaseVectorStore | 无外部依赖、逻辑透明、精确计算 | 不能扩展、无 ANN、高内存和 CPU | 保留为 Baseline |
| pgvector | 关系数据和向量统一，事务和过滤方便 | 大规模独立检索能力不如专用库，调优依赖 PostgreSQL | 需要掌握选型逻辑 |
| Qdrant | Dense / Sparse、过滤、Payload、RRF、多阶段查询，API 清晰 | 需要处理双写和索引一致性 | **当前首选实现** |
| Elasticsearch / OpenSearch | BM25、关键词搜索、聚合和向量统一 | 部署和资源成本更高 | 搜索型企业方案备选 |
| Milvus | 大规模分布式向量检索 | 当前项目过重、运维复杂 | 暂不实现 |

### 13.3 本项目决定

```text
SQL 数据库：业务事实来源
Qdrant：可重建的检索索引
```

SQL 中保存：

- Document；
- DocumentContent；
- DocumentChunk；
- ProcessingJob；
- 状态和业务元数据。

Qdrant 中保存：

```text
Point ID：稳定 chunk_id 或稳定 UUID
Dense Vector：语义向量
Sparse Vector：后续 BM25 / Sparse
Payload：
  document_id
  chunk_id
  chunk_index
  filename
  content
  content_hash
  chunk_strategy
  parser_type
  embedding_model
  后续 knowledge_base_id / tenant_id
```

### 13.4 Qdrant 开发任务

状态：`NEXT`

1. 保留现有 `VectorStore` 抽象；
2. 审查 `VectorStore` 当前接口是否同时支持写入、删除和查询；
3. 新增 `QdrantVectorStore`；
4. 增加 Qdrant 配置；
5. 创建 Collection；
6. Embedding 完成后批量 upsert；
7. 文档删除时按 `document_id` 删除 Point；
8. 支持 `document_id` Payload 过滤；
9. 建立 Payload Index；
10. 增加重建索引命令；
11. Qdrant 不可用时任务失败并可重试；
12. 用 DatabaseVectorStore 做结果对照；
13. 后续增加 Dense + Sparse Named Vector。

### 13.5 Qdrant 一致性原则

不做跨 SQL 和 Qdrant 的分布式事务。

采用：

```text
SQL 是事实来源
Qdrant 可以重建
ProcessingJob 记录索引任务
失败可重试
删除和 upsert 必须幂等
提供全量 / 单文档重建入口
```

建议未来增加：

```text
index_status
indexed_at
index_version
last_index_error
```

是否增加数据库字段需在审查当前 Document / Chunk 状态模型后决定。

---

## 十四、企业级检索评估规划

状态：`PLANNED，Qdrant 后立即进行`

### 14.1 目标

不能仅凭主观感觉说“混合检索更好”。所有检索能力必须通过同一评估集对比。

推进顺序：

```text
建立评估集
→ Dense Baseline
→ Qdrant Dense
→ BM25 / Sparse
→ RRF Hybrid
→ Reranker
→ Query Rewrite
→ 对比报告
```

### 14.2 评估集

第一阶段构建 30～50 条高质量问题，后续扩展到 100 条以上。

建议结构：

```json
{
  "query": "管理员如何重置用户密码？",
  "relevant_document_ids": [2],
  "relevant_chunk_ids": [15, 16],
  "answer_keywords": ["系统设置", "管理员"],
  "answerable": true,
  "category": "semantic",
  "difficulty": "medium"
}
```

问题类型：

```text
精确术语
错误码和 API 名称
同义表达
口语化问题
中英文混合
多段落问题
多文档问题
表格问题
无答案问题
干扰文档问题
缩写和别名
容易误召回的问题
```

### 14.3 检索指标

```text
Chunk Recall@1 / @3 / @5 / @10
Document Recall@K
Hit Rate@K
MRR@K
nDCG@K
Top-1 准确率
正确文档首次出现位置
```

### 14.4 企业业务指标

```text
无答案问题误召回率
答案可覆盖率
引用来源正确率
重复 Chunk 比例
单文档占位率
Top-K 文档多样性
错误文档干扰率
答案关键词覆盖率
```

### 14.5 性能指标

```text
Embedding 耗时
Dense 检索耗时
Sparse 检索耗时
RRF 融合耗时
Rerank 耗时
检索 P50 / P95
完整问答 P50 / P95
LLM 首 Token 延迟
```

### 14.6 报告输出

```text
evaluation_results.csv
evaluation_report.md
evaluation_config.json
```

每次实验必须记录：

```text
评估数据版本
Embedding 模型
Chunk 策略和版本
VectorStore
Top-K
Candidate-K
阈值
融合方式
Reranker
Query Rewrite 模式
指标
延迟
运行时间
Git Commit
```

---

## 十五、混合检索与多路召回规划

状态：`PLANNED`

### 15.1 第一阶段架构

```text
原始 Query
  ├─ Dense Retrieval Top-20
  └─ BM25 / Sparse Retrieval Top-20
        ↓
      RRF 融合
        ↓
      Candidate Top-20
        ↓
      Reranker
        ↓
      Final Top-5
```

### 15.2 为什么先使用 RRF

Dense Score 与 BM25 Score 不在同一分数尺度上，第一版不直接做线性加权。

使用基于排名的 RRF，优点：

- 不依赖分数归一化；
- 初始调参成本较低；
- 适合先建立可靠 Baseline；
- 方便通过评估集比较。

### 15.3 多路召回演进

第一版：

```text
Dense Content
BM25 Content
```

第二版：

```text
Dense Content
BM25 Content
Title / Heading Dense
```

第三版：

```text
原始 Query
改写 Query 1
改写 Query 2
```

所有召回结果统一转换为内部候选 DTO，再由独立 Fusion 层处理，不把融合逻辑散落在 Router 和 KnowledgeChatService 中。

---

## 十六、Reranker 规划

状态：`PLANNED`

### 16.1 目标链路

```text
Dense + Sparse 召回
→ Top-20 / Top-30 候选
→ Cross-Encoder Reranker
→ Top-5
```

### 16.2 抽象建议

```text
Reranker
├─ NoOpReranker
├─ LocalCrossEncoderReranker
└─ RemoteReranker
```

### 16.3 验证组

至少比较：

```text
Dense
Dense + BM25
Dense + BM25 + RRF
Dense + BM25 + RRF + Rerank
```

第一阶段不训练模型，先使用适合中文或多语言的现成模型，通过评估集证明收益。

---

## 十七、Query Rewrite 与高级召回规划

状态：`PLANNED，排在 Hybrid 和 Rerank 之后`

### 17.1 确定性规范化

```text
去除无意义前缀
统一产品名
缩写展开
错误码标准化
中英文别名映射
```

### 17.2 LLM Multi-Query

一个问题生成 2～3 个检索表达：

```text
原始问题
术语化表达
意图化表达
```

保护措施：

- 原始 Query 必须保留；
- 改写数量受限；
- Token 受限；
- 超时自动降级；
- 记录改写结果；
- 可配置关闭；
- 必须单独评估。

### 17.3 HyDE

作为实验模式，不作为默认主链路。

原因：假设答案可能引入不存在的事实，需要与原始 Query 并行召回并通过评估验证。

---

## 十八、企业文档解析和切片规划

状态：`PLANNED`

### 18.1 当前支持

```text
TXT
PDF
PDF OCR
recursive_character Chunk
```

### 18.2 新优先级

```text
Parent-Child Chunk
→ Markdown
→ HTML / Web
→ 表格
→ 代码
```

### 18.3 Parent-Child

推荐：

```text
Parent Chunk：
保留章节完整上下文
约 1000～2000 字符

Child Chunk：
用于向量检索
约 300～500 字符
适度 Overlap
```

链路：

```text
检索 Child
→ 找到 Parent
→ 根据策略返回 Parent 或扩展窗口
→ Rerank
→ ContextBuilder
```

数据模型候选字段：

```text
parent_chunk_id
chunk_role / chunk_level
heading_path
metadata
```

具体字段必须在审查当前 `DocumentChunk` 模型和现有迁移后确定。

### 18.4 Markdown

要求：

- 按 H1 / H2 / H3 结构切分；
- 保留标题层级；
- 代码块不从中间截断；
- 表格尽量保持完整；
- Chunk Metadata 保存 heading path。

### 18.5 HTML / 网页

要求：

- 移除 Script、Style、导航和无关元素；
- 保留标题、段落、列表、表格；
- 保存 `source_url` 和抓取时间；
- URL 导入必须考虑 SSRF；
- 不允许直接对任意内网地址发请求。

### 18.6 表格

要求：

```text
保留表头
按行组切分
子块重复表头
保留表格来源和标题
必要时生成行级文本描述
```

复杂 PDF 表格恢复暂不与基础 Parent-Child 同时实现。

### 18.7 代码

代码切片不继续使用普通字符策略。

计划使用语法树方案，例如 Tree-sitter：

```text
按文件
按 class
按 function / method
按 interface
保留 import、类名、文件路径和语言
避免函数中间截断
```

代码 RAG 作为独立里程碑开发。

---

## 十九、异步任务升级规划

### 19.1 当前实现

```text
FastAPI BackgroundTasks
→ ProcessingJobRunner
→ ProcessingJobExecutor
```

已经有完整 Job 数据模型、状态、进度、幂等和失败重试基础。

### 19.2 当前实现的限制

- 与 API 进程生命周期绑定；
- 服务重启可能丢失执行中的任务；
- 无法独立扩缩容；
- 多进程执行和任务路由能力弱；
- 重任务会与 API 争抢资源；
- 缺少标准 Retry、Timeout 和 Worker 监控。

### 19.3 推荐升级

状态：`PLANNED，P1`

```text
FastAPI
→ ProcessingJobDispatcher
→ Redis Broker
→ Celery Worker
→ ProcessingJobRunner / Executor
```

新增抽象：

```text
ProcessingJobDispatcher
├─ InProcessJobDispatcher
└─ CeleryJobDispatcher
```

消息只传：

```json
{
  "job_id": 123
}
```

Worker：

```text
收到 job_id
→ 创建独立 DB Session
→ 查询 ProcessingJob
→ 校验任务状态和幂等
→ 调用现有 Runner / Executor
→ 更新进度和状态
→ commit / rollback
→ 关闭 Session
```

### 19.4 高价值能力

```text
自动重试
指数退避
Soft / Hard Timeout
Worker 并发配置
任务幂等
stale job 恢复
失败次数
last_error
next_retry_at
Worker 心跳
任务取消
结构化日志 job_id
```

### 19.5 前置条件

在 API 和 Worker 多进程同时写数据库前，优先迁移 PostgreSQL。SQLite 不适合作为正式多进程任务数据库。

---

## 二十、Docker、PostgreSQL 与 MinIO 规划

### 20.1 Docker Compose

状态：`PLANNED，优先快速实现`

阶段一：

```text
api
qdrant
```

阶段二：

```text
api
qdrant
postgres
```

阶段三：

```text
api
worker
redis
qdrant
postgres
```

阶段四：

```text
api
worker
redis
qdrant
postgres
minio
```

最终目标：

```powershell
docker compose up -d
docker compose exec api alembic upgrade head
```

然后跑通：

```text
上传
→ 异步解析
→ Parent-Child Chunk
→ Qdrant 建库
→ Hybrid Search
→ Rerank
→ RAG 回答
→ 来源展示
```

### 20.2 PostgreSQL

状态：`PLANNED`

价值：

- 支持 API + Worker 并发；
- 更可靠的锁和事务；
- 更适合部署；
- 连接池和运维能力更成熟；
- 为后续 KnowledgeBase、权限和多租户打基础。

迁移原则：

- 继续使用 SQLAlchemy；
- 所有结构变更通过 Alembic；
- SQLite 保留本地轻量测试模式是否必要，届时再决定；
- 不同时引入 pgvector，避免与 Qdrant 主线冲突。

### 20.3 MinIO

状态：`PLANNED，快速工程增强`

现有 StorageService 抽象应扩展为：

```text
StorageService
├─ LocalStorageService
└─ MinioStorageService
```

配置：

```text
STORAGE_BACKEND=local|minio
MINIO_ENDPOINT
MINIO_ACCESS_KEY
MINIO_SECRET_KEY
MINIO_BUCKET
MINIO_SECURE
```

首版范围：

- 上传；
- 读取；
- 删除；
- exists；
- Bucket 初始化。

不做：

- 分布式集群；
- 复杂 IAM；
- 跨区域复制；
- 生命周期管理；
- 版本控制。

MinIO 预计工作量较小，可与 Docker Compose 阶段结合，但不应阻塞 Qdrant 和检索质量主线。

---

## 二十一、测试策略：简化、简化、再简化

从当前阶段起，测试目标从“覆盖每个方法”调整为：

> 用尽可能少的测试保护最关键的业务行为和架构契约。

### 21.1 必须保留

#### 完整主链路

```text
上传文档
→ 创建 full_pipeline Job
→ 解析
→ Chunk
→ Embedding
→ Qdrant Point
→ 检索
→ Knowledge Chat
→ 来源正确
```

#### VectorStore 合约

同一组核心测试验证：

```text
DatabaseVectorStore
QdrantVectorStore
```

#### Retrieval 指标

```text
Recall@K
MRR
nDCG
Baseline / Hybrid / Rerank 对比
```

#### ProcessingJob

只保留：

```text
同一文档不能有两个活动任务
成功状态流转
失败保留进度
失败后可重试
重复执行幂等
```

#### API 契约

每个核心接口通常只保留：

```text
一个成功
一个核心业务失败
一个关键参数边界
```

#### SSE

只保留：

```text
metadata → message → done 顺序
异常时 error 且无 done
取消时关闭底层流
```

### 21.2 应删除或合并

- Service、Router、Schema 对同一字段的重复断言；
- 大量只检查 Fake 收到某参数的测试；
- Pydantic 已保证的每个边界逐一测试；
- 同步和流式完全重复的业务测试；
- Repository 机械 CRUD 测试；
- 多个只差一个数值的测试；
- 对内部实现步骤过度断言；
- 对日志文案逐字匹配；
- 不会阻止真实回归的测试。

### 21.3 新测试规则

普通开发块默认最多增加：

```text
1 个成功用例
1 个核心失败用例
1 个架构边界用例
```

只有算法模块、状态机和并发模块可以增加更多必要用例。

### 21.4 当前测试技术债

- 已完成最终全量回归：`157 passed，1 skipped，1 warning`；
- 已完成独立空数据库端到端验收；
- 继续评估并逐步删除重复测试；
- Starlette TestClient / httpx 依赖弃用警告已记录，后续随依赖升级处理；
- 不追求虚高覆盖率。

---

## 二十二、版本路线重新规划

| 版本 | 核心目标 | 主要能力 | 状态 | 退出条件 |
|---|---|---|---|---|
| v0.9.0 | 面试交付版 | 文档加工、Dense 检索、同步/SSE、来源、ProcessingJob、README | `VERIFY` | 技术验收已通过，待 Git Tag / Release |
| v0.10.0 | Qdrant 检索基座 | QdrantVectorStore、Upsert/Delete、过滤、重建索引 | `NEXT` | SQL 与 Qdrant 可恢复一致 |
| v0.11.0 | 企业检索质量 | 强评估集、BM25、RRF、Reranker | `PLANNED` | 指标报告可证明改进 |
| v0.12.0 | 结构化文档 | Parent-Child、Markdown、HTML、表格基础 | `PLANNED` | 多格式文档可稳定检索 |
| v0.13.0 | 工程部署 | PostgreSQL、Celery、Redis、Docker Compose、MinIO | `PLANNED` | 一键部署，多进程可靠处理 |
| v1.0.0 | 企业级单租户 RAG | 可部署、可评估、可恢复、可观测的完整系统 | `PLANNED` | 可供小团队持续试用 |
| v1.1 | 高级检索 | Query Rewrite、Multi-Query、HyDE、代码 RAG | `PLANNED` | 高难问题指标稳定提升 |
| v1.2 | 会话和前端 | 多轮、会话模型、完整知识库 UI | `DEFERRED` | 用户可自助使用 |
| v2 | 企业 Agent | Plan / Replan、工具、审批、审计 | `DEFERRED` | 工具调用安全可追踪 |
| v3 | 平台化 | 多租户、私有化、配额计费、插件平台 | `DEFERRED` | 可规模化交付 |

---

## 二十三、面试优先开发顺序

### 里程碑 0：v0.9.0 面试交付收尾

状态：`VERIFY（技术验收完成，待 Git Tag / Release）`

```text
全量 pytest 最终验收
干净数据库端到端验证
README
架构图
固定演示文档和问题集
Git Tag / Release
```

建议工作量：0.5～1.5 个开发日。

### 里程碑 1：Qdrant

状态：`NEXT`

```text
Docker 启动 Qdrant
QdrantVectorStore
Collection 初始化
批量 Upsert
文档删除同步
Payload Filter / Index
单文档重建
全量重建
DatabaseVectorStore 对照
```

建议工作量：2～3 个开发日。

### 里程碑 2：企业检索评估

```text
30～50 条评估集
Recall@K
MRR
nDCG
无答案误召回率
重复率
文档覆盖
P50 / P95
Markdown / CSV 报告
```

建议工作量：1～2 个开发日。

### 里程碑 3：Hybrid + Rerank

```text
Dense
BM25 / Sparse
RRF
Cross-Encoder Rerank
四组对比报告
```

建议工作量：3～5 个开发日。

### 里程碑 4：Parent-Child 与 Markdown

```text
Parent / Child 数据模型
迁移
结构感知 Chunk
Markdown Parser
Child 检索 + Parent 上下文
评估对比
```

建议工作量：2～4 个开发日。

### 里程碑 5：Docker Compose + MinIO

```text
API + Qdrant
加入 MinIO
一键启动
环境变量模板
数据 Volume
```

建议工作量：1～2 个开发日。

### 里程碑 6：PostgreSQL + Celery Worker

```text
数据库迁移
Redis
Celery Worker
Dispatcher
Retry / Timeout
任务恢复
```

建议工作量：2～4 个开发日。

以上工作量为粗略工程估算，实际以当前代码复杂度为准。

---

## 二十四、当前必须完成的重要节点

### 24.1 立即验收

执行：

```powershell
pytest -v
```

记录真实结果。

来源文件名开发块只有在全量测试通过后才从 `VERIFY` 改成 `DONE`。

### 24.2 干净环境端到端

```text
删除或新建空数据库
→ alembic upgrade head
→ 启动 API
→ 上传文档
→ 创建 full_pipeline Job
→ 查询任务状态
→ 任务成功
→ Knowledge Chat
→ 来源文件名正确
→ 无答案问题拒答
```

### 24.3 README / 面试材料

至少补充：

```text
项目定位
技术栈
架构图
目录结构
启动方式
环境变量
Alembic
Tesseract
核心 API
RAG 链路
ProcessingJob
测试
评估结果
演示截图
已知限制
路线图
```

### 24.4 演示问题集

固定准备：

- 精确答案问题；
- 多来源问题；
- 指定文档问题；
- 无答案问题；
- 低相关内容过滤问题；
- ProcessingJob 失败和重试演示。

---

## 二十五、技术债分级

### 25.1 P0：当前和 v0.10 前

```text
Git Tag / Release
Qdrant 接入
Qdrant Upsert / Delete 幂等
索引重建
SQL 与 Qdrant 补偿机制
评估集升级
```

### 25.2 P1：v1.0 前

```text
PostgreSQL
Celery + Redis
Docker Compose
MinIO
Parent-Child
Markdown / HTML
结构化日志
健康检查
任务超时和恢复
storage_key
```

### 25.3 P2：v1.x

```text
Reranker 模型优化
Query Rewrite
Multi-Query
HyDE
表格和代码解析
网页采集
多轮会话
反馈闭环
文档版本和重建索引管理
复杂 OCR 清洗
```

### 25.4 P3：v2 以后

```text
认证和复杂 RBAC
多租户
Agent
Plan / Replan
企业连接器
人工审批
SaaS 配额和计费
插件平台
```

### 25.5 保留的历史技术债

```text
documents.path → storage_key
stored_name / storage_key 唯一约束
SQLite PRAGMA foreign_keys=ON 已完成，并保留回归验证
状态转换统一
DocumentContent upsert
软删除策略
分页
日志监控
缓存
备份恢复
限流和安全
Prompt Injection 防护
```

---

## 二十六、下一准确开发块：Qdrant 向量存储

状态：`NEXT`

### 26.1 本轮目标

在不破坏现有 RAG 上层链路的前提下，引入 Qdrant 作为专用向量检索实现，并保留 DatabaseVectorStore 作为 Baseline。

### 26.2 编码前必须检查的文件

若新对话无法从当前附件读取最新代码，请用户提交以下完整文件：

```text
app/services/vector_store/base.py
app/services/vector_store/database.py
app/services/embedding_service.py
app/services/embedding/base.py
app/services/embedding/factory.py
app/repositories/chunk_embedding_repository.py
app/repositories/document_chunk_repository.py
app/models/database/document_chunk.py
app/models/database/chunk_embedding.py
app/services/processing_job_executor.py 或实际 Executor 文件
app/services/document_service.py
app/core/config.py
app/main.py
requirements.txt
相关 Alembic 最新 migration
tests/test_vector_store.py
tests/test_embedding.py
tests/test_processing_job.py
```

还需要最新项目目录树，以确认：

- ProcessingJob 文件的准确名称；
- VectorStore 抽象当前是否只有 search；
- 配置工厂是否已经支持 backend 选择；
- 是否存在 Docker 目录或 Compose 文件；
- 是否存在索引重建脚本。

### 26.3 架构审查重点

```text
VectorStore 是否同时承担写入和查询
是否需要拆 VectorIndex / VectorSearcher
EmbeddingService 当前何处保存 ChunkEmbedding
Qdrant Upsert 的事务时机
Job 失败如何重试
删除文档如何同步删除 Qdrant
Point ID 如何稳定
Payload 字段和过滤
Collection 命名和模型版本
向量维度变化如何处理
是否需要多 Collection
索引重建入口
Qdrant 客户端生命周期
异常如何映射
```

### 26.4 首版最小范围

必须：

```text
Qdrant 配置
QdrantVectorStore
Collection 初始化
Dense Upsert
Dense Search
document_id Filter
按 document_id Delete
单文档重建
DatabaseVectorStore 保留
少量合约测试
Docker 启动 Qdrant
```

本轮不做：

```text
BM25 Sparse
RRF
Reranker
Query Rewrite
Parent-Child
PostgreSQL
Celery
MinIO
多租户
```

### 26.5 验收标准

- 相同 `VectorStore` 上层接口可切换 Database / Qdrant；
- 文档处理完成后 Qdrant 中有对应 Point；
- 查询可按 `document_id` 过滤；
- 删除文档后相关 Point 被删除；
- 重复 Upsert 不产生重复数据；
- Qdrant 索引可从 SQL 重建；
- Qdrant 不可用时任务进入失败状态并可重试；
- DatabaseVectorStore 仍可运行；
- 只保留关键测试。

### 26.6 推荐提交拆分

```text
1. 增加 Qdrant 配置和客户端
2. 实现 Qdrant 向量存储
3. 接入文档向量建库和删除同步
4. 增加索引重建能力
5. 增加 Docker Qdrant 和关键测试
```

推荐中文 Git 提交信息示例：

```text
增加 Qdrant 向量存储实现
接入文档向量索引与删除同步
增加 Qdrant 索引重建能力
完善 Qdrant 本地部署和关键测试
```

---

## 二十七、面试展示结构

### 27.1 三分钟介绍

```text
业务问题：企业文档难查询，新员工找资料慢
产品定位：私有知识库问答助手
核心链路：上传、解析、切片、向量化、检索、回答、来源
工程能力：分层、事务、异步任务、失败重试、SSE
质量能力：Baseline / Optimized、评估、下一步 Hybrid 和 Rerank
部署演进：Qdrant、Docker、Worker、PostgreSQL
```

### 27.2 深挖问题准备

需要能够回答：

```text
为什么不直接使用 LangChain 一把梭？
为什么 SQL 是事实来源，Qdrant 是索引？
如何处理 SQL 与 Qdrant 不一致？
为什么保留 DatabaseVectorStore？
Candidate-K 和 Top-K 的区别？
为什么业务 API 不允许传 score_threshold？
RRF 为什么适合 Dense + BM25？
Reranker 为什么只处理候选集？
Parent-Child 如何平衡召回和上下文？
BackgroundTasks 的限制是什么？
Celery 重试如何保证幂等？
SSE 中来源何时返回？
无召回为什么不调用 LLM？
来源为什么由后端生成？
```

### 27.3 简历项目表述参考

> 独立设计并实现企业私有知识库 RAG 后端，基于 FastAPI、SQLAlchemy 和 Alembic 构建文档上传、PDF/OCR 解析、结构化切片、Embedding、向量检索、受控问答及来源追溯完整链路；实现 Dense 检索、候选扩召回、去重、多文档平衡和 Baseline/Optimized 评估；设计 ProcessingJob 异步任务、幂等约束、进度管理和失败重试，并通过 SSE 提供流式回答。当前继续接入 Qdrant、混合检索、Reranker、企业级评估和 Docker 化部署。

---

## 二十八、新对话统一开场指令

在新窗口上传本文件后，可以直接发送：

> 请以这份《Knowledge Assistant 项目完整开发规划与交接文档（2026-08-04）》作为当前项目唯一事实基线。你需要继续以资深架构师、资深后端开发者、企业 RAG 架构师和面试项目导师的角色指导我。后续考虑新功能时，先检查我提供的最新项目目录和现有代码，恢复真实调用链，分析架构、职责、事务、一致性和面试价值，再进行编码。不要根据记忆重建已有文件，不要默认新建平行模块。当前项目处于 v0.9.0，技术验收已完成，待创建 Git Tag / Release，RAG、SSE、ProcessingJob 和来源追溯已经基本完成；当前主线是 Qdrant → 企业级评估 → BM25 + Vector → RRF → Reranker → Parent-Child → Docker / Worker。测试必须尽量精简，只保留关键业务、算法、并发和接口契约。请先总结你从本文档恢复的当前状态，并列出下一开发块需要检查的最新文件，然后再继续。

---

## 二十九、文档维护机制

每个小开发块只在对话中记录：

- 完成内容；
- 测试结果；
- 技术债；
- Git 提交。

每完成一个版本里程碑，再集中更新：

```text
本交接文档
README
架构图
API 说明
部署说明
评估报告
Release Notes
```

下一次必须更新本文件的节点：

```text
v0.9.0 Git Tag / Release 完成
Qdrant 接入完成
Hybrid + Rerank 完成
Parent-Child 完成
Docker + Worker 完成
v1.0.0 发布
```

---

## 三十、更新记录

### 2026-08-04（v0.9.0 技术验收与项目更名）

- 正式产品名称统一为 `Knowledge Assistant`；
- `APP_NAME` 和 README 使用统一名称；
- 完成最终全量回归：`157 passed，1 skipped，1 warning`；
- 完成独立空数据库、Alembic、文档上传和 `full_pipeline` 端到端验收；
- 完成 DocumentContent、Chunk、Embedding、Knowledge Chat、来源文件名和无答案拒答验收；
- 完成 SSE `metadata → message → done` 生命周期验收；
- 为每个 SQLite 连接开启 `PRAGMA foreign_keys=ON`；
- 验证删除 Document 后 DocumentContent、Chunk、Embedding 和 ProcessingJob 全部级联清理；
- 完成根目录 README；
- 当前仅剩 Git Tag / Release，之后进入 Qdrant 开发块。

### 2026-08-04（规划合并）

- 合并 2026-07-30 交接文档；
- 当时将版本定位为 `v0.9.0-rc1`；
- 记录非流式和流式 Knowledge Chat 已完成；
- 记录无召回拒答和不调用 LLM；
- 记录 ProcessingJob、Runner、Executor 和任务历史能力；
- 记录任务进度、并发约束和失败重试；
- 记录业务阈值不允许由外部 API 覆盖；
- 记录 `RETRIEVAL_SCORE_THRESHOLD` 与 `KNOWLEDGE_CHAT_SCORE_THRESHOLD` 的职责；
- 记录来源文件名从数据库到同步响应和 SSE metadata 的完整链路；
- 记录当时 `filename` Schema 升级及全量回归待验证；
- 将项目主线调整为面试价值优先；
- 明确 Qdrant 为当前首选向量数据库实现；
- 增加企业级检索评估方案；
- 增加 BM25、RRF、Reranker 和 Query Rewrite 路线；
- 增加 Parent-Child、Markdown、HTML、表格和代码切片路线；
- 增加 PostgreSQL、Celery、Redis、Docker Compose 和 MinIO 路线；
- 大幅简化测试策略；
- 明确下一准确开发块为 Qdrant 向量存储接入。

---

## 三十一、当前最终状态摘要

```text
项目：Knowledge Assistant
版本：v0.9.0
发布状态：技术验收已完成，待创建 Git Tag / Release
阶段：可面试展示的后端 RAG MVP

已完成：
文档上传 / 删除
TXT / PDF / OCR
Chunk V1
Embedding
SQL 向量持久化
Dense Retrieval
Baseline / Optimized
ContextBuilder
Knowledge Chat
SSE
真实来源和文件名
ProcessingJob
Alembic
测试隔离
检索评估框架

技术验收：
全量 pytest：157 passed，1 skipped，1 warning
独立空数据库 + Alembic：通过
文档上传 + full_pipeline：通过
DocumentContent / Chunk / Embedding：通过
Knowledge Chat + 来源文件名：通过
无答案拒答：通过
SSE 生命周期：通过
SQLite foreign_keys=ON：通过
文档及 ProcessingJob 删除级联：通过
README：已完成
Git Tag / Release：待创建

下一主线：
Qdrant
→ 企业级评估
→ BM25 + Vector
→ RRF
→ Reranker
→ Parent-Child
→ Docker / PostgreSQL / Celery / MinIO

明确延后：
多租户
复杂权限
Agent
Plan / Replan
完整管理后台
平台化和商业化
```

---

## 三十二、v0.9.0 发布门禁

已通过：

```text
全量测试
干净数据库迁移
核心 RAG E2E
同步问答
SSE 问答
来源追溯
无答案拒答
SQLite 外键
删除级联
README
```

待执行：

```text
检查 git status / git diff
提交发布文档
创建 v0.9.0 Git Tag
创建 Release Notes
```

上述发布动作完成后，下一准确开发块为 `v0.10.0 Qdrant 向量存储`。
