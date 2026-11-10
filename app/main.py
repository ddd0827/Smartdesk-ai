from __future__ import annotations

import base64
import binascii
import logging
import secrets
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.api import chat, documents, feedback, health, meta, mock_itsm, sessions, stats
from app.config import Settings
from app.deps import build_container
from app.llm.base import LLMClient
from app.logging_conf import configure_logging

logger = logging.getLogger(__name__)

_AUTH_EXEMPT_PATHS = {"/api/health", "/favicon.ico"}
_WWW_AUTH_HEADER = {"WWW-Authenticate": 'Basic realm="SmartDesk", charset="UTF-8"'}


def _basic_auth_ok(request: Request, settings: Settings) -> bool:
    """Optional HTTP Basic gate for public demo deployments.

    Disabled when ACCESS_PASSWORD is empty. Health checks stay unauthenticated
    so Docker/load balancers can probe the service.
    """
    expected = settings.access_password_value
    if not expected:
        return True
    header = request.headers.get("authorization", "")
    if not header.lower().startswith("basic "):
        return False
    encoded = header.split(" ", 1)[1].strip()
    try:
        decoded = base64.b64decode(encoded, validate=True).decode("utf-8")
    except (ValueError, UnicodeDecodeError, binascii.Error):
        return False
    username, separator, password = decoded.partition(":")
    if not separator:
        return False
    return secrets.compare_digest(
        username, settings.access_username
    ) and secrets.compare_digest(password, expected)


def create_app(
    settings: Settings | None = None, *, llm: LLMClient | None = None
) -> FastAPI:
    settings = settings or Settings()
    configure_logging(
        logging.WARNING if settings.app_env == "test" else logging.INFO
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        container = await build_container(settings, llm=llm)
        app.state.container = container
        logger.info(
            "SmartDesk 已启动 env=%s mode=%s model=%s embedding=%s",
            settings.app_env,
            "mock" if settings.is_mock_llm else settings.llm_provider,
            container.llm.model,
            container.embedder.signature,
        )
        try:
            yield
        finally:
            await container.close()
            logger.info("SmartDesk 已关闭")

    app = FastAPI(
        title=settings.app_name,
        description="悦家售后客服 AI 智能体（RAG + 工具调用 + FastAPI）",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.container = None

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        request.state.request_id = request_id
        if (
            request.url.path not in _AUTH_EXEMPT_PATHS
            and not _basic_auth_ok(request, settings)
        ):
            response = Response(
                status_code=401,
                content="需要访问密码",
                media_type="text/plain; charset=utf-8",
                headers=_WWW_AUTH_HEADER,
            )
            response.headers["X-Request-ID"] = request_id
            logger.info(
                "%s %s -> 401 auth required rid=%s",
                request.method,
                request.url.path,
                request_id,
            )
            return response
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception(
                "未处理异常 method=%s path=%s rid=%s",
                request.method,
                request.url.path,
                request_id,
            )
            raise
        duration_ms = (time.perf_counter() - started) * 1000
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "%s %s -> %s %.0fms rid=%s",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            request_id,
        )
        return response

    @app.exception_handler(Exception)
    async def unhandled_exception(request: Request, exc: Exception):
        logger.exception("未捕获异常 rid=%s", getattr(request.state, "request_id", "-"))
        return JSONResponse(
            status_code=500,
            content={
                "detail": "服务器内部错误",
                "request_id": getattr(request.state, "request_id", None),
            },
        )

    for router in (
        health.router,
        meta.router,
        stats.router,
        chat.router,
        sessions.router,
        documents.router,
        feedback.router,
        mock_itsm.router,
    ):
        app.include_router(router, prefix="/api")

    app.mount(
        "/static",
        StaticFiles(directory=str(settings.web_dir)),
        name="static",
    )

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(settings.web_dir / "index.html")

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> Response:
        return Response(status_code=204)

    return app


app = create_app()


__all__ = ["app", "create_app"]
