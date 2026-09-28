from typing import Annotated

from fastapi import APIRouter, Depends

from visionguard.api.dependencies import get_inference_service
from visionguard.api.services.inference import InferenceService

router = APIRouter(tags=["service"])


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
def ready(
    service: Annotated[InferenceService, Depends(get_inference_service)],
) -> dict[str, object]:
    return {"ready": service.ready, "detail": service.error}
