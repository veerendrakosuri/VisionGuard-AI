"""FastAPI dependencies."""

import secrets
from typing import Annotated, cast

from fastapi import Depends, Request
from fastapi.security import APIKeyHeader

from visionguard.api.errors import ApiError
from visionguard.api.services.inference import InferenceService

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_inference_service(request: Request) -> InferenceService:
    return cast(InferenceService, request.app.state.inference_service)


def require_api_key(
    request: Request,
    supplied: Annotated[str | None, Depends(api_key_header)],
) -> None:
    expected = get_inference_service(request).config.security.api_key
    if expected is not None and (
        supplied is None or not secrets.compare_digest(supplied, expected)
    ):
        raise ApiError(401, "invalid_api_key", "A valid X-API-Key header is required")
