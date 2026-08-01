from fastapi import FastAPI
from fastapi.responses import JSONResponse
from app.core.config import get_settings
from app.api.chat import router as chat_router
from app.api.knowledge import router as knowledge_router
from app.api.retrieval import router as retrieval_router
from app.api.knowledge_chat import router as knowledge_chat_router

from app.models.database import *


settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    debug=settings.debug,
)

app.include_router(chat_router)
app.include_router(knowledge_router)
app.include_router(retrieval_router)
app.include_router(knowledge_chat_router)

@app.get("/")
def read_root():
    return {
        "status": "ok",
        "app_name": settings.app_name,
        "model": settings.model_name,
    }
