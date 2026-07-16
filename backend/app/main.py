from fastapi import FastAPI
from fastapi.responses import JSONResponse
from app.core.config import get_settings
from app.api.chat import router as chat_router

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    debug=settings.debug,
)

app.include_router(chat_router)

@app.get("/")
def read_root():
    return {
        "status": "ok",
        "app_name": settings.app_name,
        "model": settings.model_name,
    }
