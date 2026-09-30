"""Thread-safe, checkpoint-backed PatchCore inference service."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from visionguard.api.errors import ApiError
from visionguard.api.models import InspectionResult, ModelInfo
from visionguard.api.persistence import InspectionRepository
from visionguard.cli import _summarize
from visionguard.config import AppConfig
from visionguard.pipeline import load_trained_components
from visionguard.visualization import save_anomaly_overlay


class InferenceService:
    def __init__(
        self, config: AppConfig, repository: InspectionRepository | None = None
    ) -> None:
        self.config = config
        self.output_dir = config.api.output_dir
        self.checkpoint: Path | None = None
        self.model: Any = None
        self.engine: Any = None
        self.ready = False
        self.error: str | None = None
        self._lock = threading.Lock()
        self.repository = repository

    def initialize(self) -> None:
        """Load model weights once for reuse by all requests."""
        try:
            if self.repository is not None:
                self.repository.initialize()
                self.repository.delete_expired(self.config.database.retention_days)
            checkpoint, self.model, self.engine = load_trained_components(self.config)
            self.checkpoint = checkpoint
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.ready = True
        except Exception as exc:
            self.error = str(exc)
            self.ready = False

    def inspect(self, image_path: Path, original_filename: str) -> InspectionResult:
        if not self.ready:
            raise ApiError(503, "model_not_ready", self.error or "Model is not ready")
        inspection_id = str(uuid4())
        inspection_dir = self.output_dir / inspection_id
        inspection_dir.mkdir(parents=True, exist_ok=False)
        started = time.perf_counter()
        try:
            with self._lock:
                predictions = self.engine.predict(
                    model=self.model, data_path=image_path, return_predictions=True
                )
            elapsed_ms = (time.perf_counter() - started) * 1000
            summary = _summarize(predictions, self.config.decision.threshold)
            batch = predictions[0] if isinstance(predictions, list) else predictions
            anomaly_map = getattr(batch, "anomaly_map", None)
            if anomaly_map is None:
                raise RuntimeError("Model did not return an anomaly map")
            save_anomaly_overlay(image_path, anomaly_map, inspection_dir / "heatmap.png")
            configured = self.config.decision.threshold is not None
            result = InspectionResult(
                inspection_id=inspection_id,
                filename=original_filename,
                decision=summary["decision"],
                anomaly_score=float(summary["pred_score"]),
                threshold=self.config.decision.threshold,
                threshold_source="configured" if configured else "model_provisional",
                inference_time_ms=round(elapsed_ms, 2),
                heatmap_url=f"/api/v1/inspections/{inspection_id}/heatmap",
                model=ModelInfo(
                    category=self.config.data.category, backbone=self.config.model.backbone
                ),
            )
            (inspection_dir / "result.json").write_text(
                result.model_dump_json(indent=2), encoding="utf-8"
            )
            if self.repository is not None:
                self.repository.save(result)
            return result
        except ApiError:
            raise
        except Exception as exc:
            raise ApiError(500, "inference_failed", "Model inference failed") from exc
        finally:
            image_path.unlink(missing_ok=True)

    def get(self, inspection_id: str) -> InspectionResult:
        if self.repository is not None:
            stored = self.repository.get(inspection_id)
            if stored is not None:
                return stored
        result = self.output_dir / inspection_id / "result.json"
        if not result.is_file():
            raise ApiError(404, "inspection_not_found", "Inspection was not found")
        return cast(
            InspectionResult,
            InspectionResult.model_validate(json.loads(result.read_text(encoding="utf-8"))),
        )

    def heatmap(self, inspection_id: str) -> Path:
        path = self.output_dir / inspection_id / "heatmap.png"
        if not path.is_file():
            raise ApiError(404, "heatmap_not_found", "Heatmap was not found")
        return path

    def list(
        self, limit: int, offset: int, decision: str | None
    ) -> tuple[list[InspectionResult], int]:
        if self.repository is None:
            raise ApiError(503, "database_not_ready", "Inspection database is unavailable")
        if decision not in (None, "PASS", "FAIL"):
            raise ApiError(400, "invalid_decision", "Decision must be PASS or FAIL")
        from typing import Literal, cast

        return self.repository.list(
            limit, offset, cast(Literal["PASS", "FAIL"] | None, decision)
        )
