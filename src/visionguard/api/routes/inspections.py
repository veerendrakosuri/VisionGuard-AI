"""Inspection upload and retrieval routes."""

from pathlib import Path
from typing import Annotated
from uuid import uuid4

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import FileResponse

from visionguard.api.dependencies import get_inference_service
from visionguard.api.errors import ApiError
from visionguard.api.models import AnalyticsSummary, InspectionPage, InspectionResult
from visionguard.api.services.inference import InferenceService

router = APIRouter(prefix="/api/v1/inspections", tags=["inspections"])
MIME_TYPES = {"image/png", "image/jpeg"}


@router.post("", response_model=InspectionResult, status_code=201)
def create_inspection(
    image: Annotated[UploadFile, File(description="PNG or JPEG industrial image")],
    service: Annotated[InferenceService, Depends(get_inference_service)],
) -> InspectionResult:
    filename = image.filename or ""
    if not filename or Path(filename).name != filename:
        raise ApiError(400, "unsafe_filename", "Filename is missing or unsafe")
    extension = Path(filename).suffix.lower()
    if (
        extension not in service.config.api.allowed_extensions
        or image.content_type not in MIME_TYPES
    ):
        raise ApiError(415, "unsupported_image", "Only configured PNG and JPEG images are allowed")
    payload = image.file.read(service.config.api.max_upload_bytes + 1)
    if not payload:
        raise ApiError(400, "empty_upload", "Uploaded image is empty")
    if len(payload) > service.config.api.max_upload_bytes:
        raise ApiError(413, "upload_too_large", "Uploaded image exceeds the configured limit")
    decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise ApiError(400, "invalid_image", "Uploaded file is not a decodable image")
    service.output_dir.mkdir(parents=True, exist_ok=True)
    temporary = service.output_dir / f"upload-{uuid4()}{extension}"
    temporary.write_bytes(payload)
    return service.inspect(temporary, filename)


@router.get("", response_model=InspectionPage)
def list_inspections(
    service: Annotated[InferenceService, Depends(get_inference_service)],
    limit: int = 20,
    offset: int = 0,
    decision: str | None = None,
) -> InspectionPage:
    if not 1 <= limit <= 100 or offset < 0:
        raise ApiError(400, "invalid_pagination", "Limit must be 1-100 and offset non-negative")
    items, total = service.list(limit, offset, decision)
    return InspectionPage(items=items, total=total, limit=limit, offset=offset)


@router.get("/analytics/summary", response_model=AnalyticsSummary)
def analytics_summary(
    service: Annotated[InferenceService, Depends(get_inference_service)],
) -> AnalyticsSummary:
    if service.repository is None:
        raise ApiError(503, "database_not_ready", "Inspection database is unavailable")
    return service.repository.analytics()


@router.get("/{inspection_id}", response_model=InspectionResult)
def get_inspection(
    inspection_id: str,
    service: Annotated[InferenceService, Depends(get_inference_service)],
) -> InspectionResult:
    return service.get(inspection_id)


@router.get("/{inspection_id}/heatmap", response_class=FileResponse)
def get_heatmap(
    inspection_id: str,
    service: Annotated[InferenceService, Depends(get_inference_service)],
) -> FileResponse:
    return FileResponse(service.heatmap(inspection_id), media_type="image/png")
