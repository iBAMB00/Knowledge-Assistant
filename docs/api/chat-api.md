# Chat API

## 普通聊天

POST

/chat

Request

{
    "message":"你好"
}

Response

{
    "answer":"你好"
}

---

## 流式聊天

POST

/chat/stream

Content-Type

text/event-stream

Response

event: message

data:
{
    "content":"你好"
}

event: done

data:{}