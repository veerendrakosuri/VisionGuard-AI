"""FastAPI dependencies."""

from typing import cast

from fastapi import Request

from visionguard.api.services.inference import InferenceService


def get_inference_service(request: Request) -> InferenceService:
    return cast(InferenceService, request.app.state.inference_service)
