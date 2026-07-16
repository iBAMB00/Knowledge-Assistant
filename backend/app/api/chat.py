from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.llm_service import LLMService
from fastapi.responses import StreamingResponse
import json
from collections.abc import Iterator


router = APIRouter()

llm_service = LLMService()

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    answer: str

@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        answer = llm_service.chat(request.message)
        return ChatResponse(answer=answer)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="模型调用失败"
        )
    
@router.post("/chat/stream")
def stream_chat(request: ChatRequest) -> StreamingResponse:
    if not request.message or not request.message.strip():
        raise HTTPException(
            status_code=400,
            detail="message cannot be empty"
        )
    try:
        return StreamingResponse(
            generate_sse(request.message),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            }
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="模型调用失败"
        )


def generate_sse(message: str) -> Iterator[str]:
    try:
        for content in llm_service.stream_chat(message):
            if not content:
                continue

            data = json.dumps(
                {"content": content},
                ensure_ascii=False,
            )

            yield f"event: message\ndata: {data}\n\n"

        yield "event: done\ndata: {}\n\n"

    except ValueError as exc:
        data = json.dumps(
            {"message": str(exc)},
            ensure_ascii=False,
        )
        yield f"event: error\ndata: {data}\n\n"

    except Exception:
        data = json.dumps(
            {"message": "模型调用失败"},
            ensure_ascii=False,
        )
        yield f"event: error\ndata: {data}\n\n"
