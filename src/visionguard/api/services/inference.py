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
from visionguard.cli import _summarize
from visionguard.config import AppConfig
from visionguard.pipeline import find_checkpoint, require_anomalib
from visionguard.visualization import save_anomaly_overlay


class InferenceService:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.output_dir = config.api.output_dir
        self.checkpoint: Path | None = None
        self.model: Any = None
        self.engine: Any = None
        self.ready = False
        self.error: str | None = None
        self._lock = threading.Lock()

    def initialize(self) -> None:
        """Load model weights once for reuse by all requests."""
        try:
            checkpoint = self.config.api.checkpoint or find_checkpoint(
                self.config.runtime.output_dir
            )
            _, engine_cls, patchcore_cls = require_anomalib()
            import anomalib
            from torch.serialization import safe_globals

            # The checkpoint was produced locally by Week 1. PyTorch 2.6+ requires
            # Anomalib's precision enum to be explicitly allowlisted during loading.
            with safe_globals([anomalib.PrecisionType]):
                self.model = patchcore_cls.load_from_checkpoint(str(checkpoint))
            self.engine = engine_cls(
                accelerator=self.config.runtime.accelerator,
                devices=self.config.runtime.devices,
                default_root_dir=self.config.runtime.output_dir,
            )
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
            return result
        except ApiError:
            raise
        except Exception as exc:
            raise ApiError(500, "inference_failed", "Model inference failed") from exc
        finally:
            image_path.unlink(missing_ok=True)

    def get(self, inspection_id: str) -> InspectionResult:
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
