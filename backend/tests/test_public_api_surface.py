"""v1.0 对外 API 面冻结测试。"""

from fastapi import FastAPI
from fastapi.routing import iter_route_contexts
from fastapi.testclient import TestClient

from app.main import app


INTERNAL_DEBUG_PATHS = {
    "/documents/{document_id}/process",
    "/documents/{document_id}/content",
    "/documents/{document_id}/chunks",
    "/documents/{document_id}/chunk-summary",
    "/documents/{document_id}/embeddings",
    "/knowledge/retrieval/debug",
}

PUBLIC_PRODUCT_PATHS = {
    "/auth/register",
    "/auth/login",
    "/auth/me",
    "/knowledge-bases/",
    "/knowledge-bases/{knowledge_base_id}",
    "/documents/",
    "/documents/{document_id}",
    "/documents/{document_id}/processing-jobs",
    "/documents/{document_id}/processing-jobs/latest",
    "/processing-jobs/{job_id}",
    "/knowledge/chat",
    "/knowledge/chat/stream",
    "/agent/chat",
    "/agent/chat/stream",
    "/agent/runtimes",
    "/agent/runs",
    "/agent/runs/{agent_run_id}",
    "/health",
    "/health/ready",
}


def _registered_api_paths(application: FastAPI) -> set[str]:
    """返回应用实际注册的 HTTP 路由路径。"""
    return {
        route_context.path
        for route_context in iter_route_contexts(application.routes)
    }


def test_generic_chat_routes_are_not_registered() -> None:
    """v1.0 正式应用不再提供匿名通用模型代理。"""
    registered_paths = _registered_api_paths(app)

    assert "/chat" not in registered_paths
    assert "/chat/stream" not in registered_paths


def test_internal_debug_routes_remain_registered_but_hidden() -> None:
    """内部排障接口保留可调用能力，但不进入公开 OpenAPI。"""
    registered_paths = _registered_api_paths(app)
    openapi_paths = set(app.openapi()["paths"])

    assert INTERNAL_DEBUG_PATHS <= registered_paths
    assert INTERNAL_DEBUG_PATHS.isdisjoint(openapi_paths)


def test_public_product_routes_remain_in_openapi() -> None:
    """前端依赖的正式产品接口必须持续出现在 OpenAPI。"""
    openapi_paths = set(app.openapi()["paths"])

    assert PUBLIC_PRODUCT_PATHS <= openapi_paths


def test_root_response_does_not_disclose_model_name() -> None:
    """匿名根接口只返回应用级状态，不暴露底层模型配置。"""
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200

    payload = response.json()

    assert payload["status"] == "ok"
    assert "app_name" in payload
    assert "model" not in payload
    assert "model_name" not in payload