from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.auth import router as auth_router
from app.api.health import router as health_router
from app.api.knowledge import router as knowledge_router
from app.api.knowledge_base import router as knowledge_base_router
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

if settings.cors_allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allowed_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=[
            "GET",
            "POST",
            "PATCH",
            "DELETE",
            "OPTIONS",
        ],
        allow_headers=[
            "Authorization",
            "Content-Type",
            "X-Request-ID",
        ],
        expose_headers=["X-Request-ID"],
    )

app.add_middleware(RequestContextMiddleware)

app.include_router(health_router)
app.include_router(auth_router)
app.include_router(knowledge_base_router)
app.include_router(knowledge_router)
app.include_router(retrieval_router)
app.include_router(knowledge_chat_router)
app.include_router(processing_job_router)


@app.get("/")
def read_root() -> dict[str, str]:
    """返回匿名应用状态，不暴露底层模型配置。"""
    return {
        "status": "ok",
        "app_name": settings.app_name,
    }