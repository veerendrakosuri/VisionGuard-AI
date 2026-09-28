"""API response schemas."""

from typing import Literal

from pydantic import BaseModel


class ModelInfo(BaseModel):  # type: ignore[misc]
    name: str = "PatchCore"
    category: str
    backbone: str


class InspectionResult(BaseModel):  # type: ignore[misc]
    inspection_id: str
    filename: str
    status: Literal["completed"] = "completed"
    decision: Literal["PASS", "FAIL"]
    anomaly_score: float
    threshold: float | None
    threshold_source: Literal["configured", "model_provisional"]
    inference_time_ms: float
    heatmap_url: str
    model: ModelInfo


class ErrorDetail(BaseModel):  # type: ignore[misc]
    code: str
    message: str
    request_id: str | None = None


class ErrorResponse(BaseModel):  # type: ignore[misc]
    error: ErrorDetail
