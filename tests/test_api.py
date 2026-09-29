from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from visionguard.api.app import create_app
from visionguard.api.errors import ApiError
from visionguard.api.models import InspectionResult, ModelInfo
from visionguard.api.services.inference import InferenceService
from visionguard.config import AppConfig, load_config


class FakeInferenceService:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.output_dir = config.api.output_dir
        self.ready = True
        self.error: str | None = None
        self.results: dict[str, InspectionResult] = {}

    def inspect(self, image_path: Path, original_filename: str) -> InspectionResult:
        assert image_path.is_file()
        result = InspectionResult(
            inspection_id="test-inspection",
            filename=original_filename,
            decision="PASS",
            anomaly_score=0.12,
            threshold=None,
            threshold_source="model_provisional",
            inference_time_ms=4.2,
            heatmap_url="/api/v1/inspections/test-inspection/heatmap",
            model=ModelInfo(category="sheet_metal", backbone="wide_resnet50_2"),
        )
        directory = self.output_dir / result.inspection_id
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "heatmap.png").write_bytes(b"png")
        self.results[result.inspection_id] = result
        image_path.unlink()
        return result

    def get(self, inspection_id: str) -> InspectionResult:
        try:
            return self.results[inspection_id]
        except KeyError as exc:
            raise ApiError(404, "inspection_not_found", "Inspection was not found") from exc

    def heatmap(self, inspection_id: str) -> Path:
        path = self.output_dir / inspection_id / "heatmap.png"
        if not path.exists():
            raise ApiError(404, "heatmap_not_found", "Heatmap was not found")
        return path


@pytest.fixture
def api_config(tmp_path: Path) -> AppConfig:
    config = load_config("configs/patchcore_mvtecad2.yaml")
    return replace(
        config,
        api=replace(
            config.api, output_dir=tmp_path / "inspections", max_upload_bytes=1000
        ),
    )


@pytest.fixture
def client(api_config: AppConfig) -> TestClient:
    service = FakeInferenceService(api_config)
    app = create_app(api_config, cast(Any, service))
    return TestClient(app)


def png_bytes() -> bytes:
    success, encoded = cv2.imencode(".png", np.zeros((8, 8, 3), dtype=np.uint8))
    assert success
    return bytes(encoded)


def test_health_and_readiness(client: TestClient) -> None:
    with client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/ready").json() == {"ready": True, "detail": None}
        metrics = client.get("/metrics")
        assert metrics.status_code == 200
        assert "visionguard_model_ready" in metrics.text


def test_upload_lookup_and_heatmap(client: TestClient) -> None:
    with client:
        response = client.post(
            "/api/v1/inspections", files={"image": ("part.png", png_bytes(), "image/png")}
        )
        assert response.status_code == 201
        assert response.json()["decision"] == "PASS"
        assert response.json()["threshold_source"] == "model_provisional"
        inspection_id = response.json()["inspection_id"]
        assert client.get(f"/api/v1/inspections/{inspection_id}").status_code == 200
        assert client.get(f"/api/v1/inspections/{inspection_id}/heatmap").status_code == 200


@pytest.mark.parametrize(
    ("filename", "content", "mime", "status", "code"),
    [
        ("part.gif", b"GIF89a", "image/gif", 415, "unsupported_image"),
        ("part.png", b"", "image/png", 400, "empty_upload"),
        ("part.png", b"not-an-image", "image/png", 400, "invalid_image"),
        ("../part.png", png_bytes(), "image/png", 400, "unsafe_filename"),
        ("part.png", b"x" * 1001, "image/png", 413, "upload_too_large"),
    ],
)
def test_invalid_uploads(
    client: TestClient,
    filename: str,
    content: bytes,
    mime: str,
    status: int,
    code: str,
) -> None:
    with client:
        response = client.post(
            "/api/v1/inspections", files={"image": (filename, content, mime)}
        )
        assert response.status_code == status
        assert response.json()["error"]["code"] == code


def test_missing_inspection(client: TestClient) -> None:
    with client:
        response = client.get("/api/v1/inspections/missing")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "inspection_not_found"


def test_missing_checkpoint_is_reported(api_config: AppConfig, tmp_path: Path) -> None:
    config = replace(
        api_config,
        runtime=replace(api_config.runtime, output_dir=tmp_path / "no-artifacts"),
        api=replace(api_config.api, checkpoint=None),
    )
    service = InferenceService(config)
    service.initialize()
    assert service.ready is False
    assert service.error is not None


def test_inference_failure_is_safe_and_cleans_upload(
    api_config: AppConfig, tmp_path: Path
) -> None:
    class FailingEngine:
        def predict(self, **_: object) -> None:
            raise RuntimeError("internal model detail")

    service = InferenceService(api_config)
    service.ready = True
    service.engine = FailingEngine()
    service.model = object()
    upload = tmp_path / "sample.png"
    upload.write_bytes(png_bytes())
    with pytest.raises(ApiError, match="Model inference failed") as error:
        service.inspect(upload, "sample.png")
    assert error.value.status_code == 500
    assert not upload.exists()


def test_api_key_protects_inspection_routes(api_config: AppConfig) -> None:
    protected = replace(
        api_config, security=replace(api_config.security, api_key="test-secret")
    )
    service = FakeInferenceService(protected)
    with TestClient(create_app(protected, cast(Any, service))) as protected_client:
        denied = protected_client.get("/api/v1/inspections/missing")
        allowed = protected_client.get(
            "/api/v1/inspections/missing", headers={"X-API-Key": "test-secret"}
        )
    assert denied.status_code == 401
    assert denied.json()["error"]["code"] == "invalid_api_key"
    assert allowed.status_code == 404


def test_rate_limit_returns_retry_header(api_config: AppConfig) -> None:
    limited = replace(
        api_config,
        security=replace(api_config.security, rate_limit_per_minute=1),
    )
    service = FakeInferenceService(limited)
    with TestClient(create_app(limited, cast(Any, service))) as limited_client:
        first = limited_client.get("/api/v1/inspections/missing")
        second = limited_client.get("/api/v1/inspections/missing")
    assert first.status_code == 404
    assert second.status_code == 429
    assert second.headers["Retry-After"] == "60"
