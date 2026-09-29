"""FastAPI application factory."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from visionguard import __version__
from visionguard.api.errors import ApiError, api_error_handler
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

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if service is None:
            inference.initialize()
        app.state.inference_service = inference
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
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    app.add_exception_handler(ApiError, api_error_handler)
    app.include_router(health_router)
    app.include_router(inspections_router)
    return app
