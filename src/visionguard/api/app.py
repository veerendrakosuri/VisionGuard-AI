"""FastAPI application factory."""

from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from uuid import uuid4

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from visionguard import __version__
from visionguard.api.errors import ApiError, api_error_handler
from visionguard.api.monitoring import ApiMetrics
from visionguard.api.persistence import InspectionRepository
from visionguard.api.routes.health import router as health_router
from visionguard.api.routes.inspections import router as inspections_router
from visionguard.api.services.inference import InferenceService
from visionguard.config import AppConfig, load_config


def create_app(config: AppConfig | None = None, service: InferenceService | None = None) -> FastAPI:
    settings = config or load_config(
        Path(os.getenv("VISIONGUARD_CONFIG", "configs/patchcore_mvtecad2.yaml"))
    )
    repository = InspectionRepository(settings.database.url) if service is None else None
    inference = service or InferenceService(settings, repository)
    api_metrics = ApiMetrics()
    rate_windows: dict[str, deque[float]] = defaultdict(deque)
    rate_lock = Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if service is None:
            inference.initialize()
        api_metrics.model_ready.set(1 if inference.ready else 0)
        app.state.inference_service = inference
        app.state.metrics = api_metrics
        yield

    app = FastAPI(
        title="VisionGuard AI API",
        version=__version__,
        description=(
            "Industrial visual anomaly detection. Decisions are provisional until calibrated."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.api.allowed_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_id(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request.state.request_id = request.headers.get("X-Request-ID", str(uuid4()))
        started = time.perf_counter()
        client = request.client.host if request.client else "unknown"
        if request.url.path.startswith("/api/"):
            now = time.monotonic()
            with rate_lock:
                window = rate_windows[client]
                while window and window[0] <= now - 60:
                    window.popleft()
                if len(window) >= settings.security.rate_limit_per_minute:
                    response = JSONResponse(
                        status_code=429,
                        content={
                            "error": {
                                "code": "rate_limit_exceeded",
                                "message": "Too many requests; retry after one minute",
                                "request_id": request.state.request_id,
                            }
                        },
                        headers={"Retry-After": "60"},
                    )
                    api_metrics.requests.labels(
                        request.method, request.url.path, "429"
                    ).inc()
                    response.headers["X-Request-ID"] = request.state.request_id
                    return response
                window.append(now)
        response = await call_next(request)
        route = request.scope.get("route")
        metric_path = getattr(route, "path", request.url.path)
        api_metrics.requests.labels(
            request.method, metric_path, str(response.status_code)
        ).inc()
        api_metrics.latency.labels(request.method, metric_path).observe(
            time.perf_counter() - started
        )
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    app.add_exception_handler(ApiError, api_error_handler)
    app.include_router(health_router)
    app.include_router(inspections_router)
    return app
