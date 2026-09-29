"""API response schemas."""

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field


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
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class InspectionPage(BaseModel):  # type: ignore[misc]
    items: list[InspectionResult]
    total: int
    limit: int
    offset: int


class AnalyticsSummary(BaseModel):  # type: ignore[misc]
    total_inspections: int
    pass_count: int
    fail_count: int
    fail_rate: float
    average_anomaly_score: float
    average_inference_time_ms: float


class ErrorDetail(BaseModel):  # type: ignore[misc]
    code: str
    message: str
    request_id: str | None = None


class ErrorResponse(BaseModel):  # type: ignore[misc]
    error: ErrorDetail
