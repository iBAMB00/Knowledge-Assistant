# Streaming Chat Architecture

## 架构

Client

↓

HTTP POST

↓

FastAPI Router

↓

LLMService

↓

OpenAI Compatible API

↓

StreamingResponse

↓

SSE Event

↓

Client

---

## 为什么采用SSE？

相比普通HTTP：

HTTP需要等待整个回答结束。

SSE可以做到：

模型生成一个Token

↓

立即发送

↓

用户立即看到

提升用户体验。

---

## StreamingResponse

FastAPI提供的流式响应对象。

核心思想：

Generator

↓

yield

↓

Chunk

↓

StreamingResponse

↓

HTTP Chunk

↓

Client

---

## 为什么使用Generator？

Generator不会一次生成所有内容。

而是：

需要一个

生成一个

发送一个

因此内存占用更低。

也更符合LLM输出方式。