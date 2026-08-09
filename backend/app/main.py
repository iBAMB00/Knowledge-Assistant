from fastapi import FastAPI

from app.api.auth import router as auth_router
from app.api.chat import router as chat_router
from app.api.health import router as health_router
from app.api.knowledge import router as knowledge_router
from app.api.knowledge_chat import router as knowledge_chat_router
from app.api.processing_job import router as processing_job_router
from app.api.retrieval import router as retrieval_router
from app.core.config import get_settings
from app.core.logging_config import configure_application_logging
from app.middleware.request_context import RequestContextMiddleware
from app.models.database import *


settings = get_settings()
configure_application_logging(settings.log_level)

app = FastAPI(
    title=settings.app_name,
    debug=settings.debug,
)

app.add_middleware(RequestContextMiddleware)

app.include_router(health_router)
app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(knowledge_router)
app.include_router(retrieval_router)
app.include_router(knowledge_chat_router)
app.include_router(processing_job_router)

@app.get("/")
def read_root():
    return {
        "status": "ok",
        "app_name": settings.app_name,
        "model": settings.model_name,
    }
