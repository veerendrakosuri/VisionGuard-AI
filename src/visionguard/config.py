"""Typed configuration loading and validation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    root: Path
    category: str
    image_size: tuple[int, int]
    train_batch_size: int
    eval_batch_size: int
    num_workers: int


@dataclass(frozen=True)
class ModelConfig:
    backbone: str
    layers: tuple[str, ...]
    coreset_sampling_ratio: float
    num_neighbors: int


@dataclass(frozen=True)
class RuntimeConfig:
    accelerator: str
    devices: int
    output_dir: Path
    log_level: str


@dataclass(frozen=True)
class DecisionConfig:
    threshold: float | None


@dataclass(frozen=True)
class ApiConfig:
    host: str
    port: int
    allowed_origins: tuple[str, ...]
    max_upload_bytes: int
    allowed_extensions: tuple[str, ...]
    output_dir: Path
    checkpoint: Path | None


@dataclass(frozen=True)
class AppConfig:
    name: str
    seed: int
    data: DataConfig
    model: ModelConfig
    runtime: RuntimeConfig
    decision: DecisionConfig
    api: ApiConfig


def _required(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing required configuration key: {key}")
    return mapping[key]


def load_config(path: str | Path) -> AppConfig:
    """Load and validate a VisionGuard YAML configuration file."""
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a mapping")

    project = _required(raw, "project")
    data = _required(raw, "data")
    model = _required(raw, "model")
    runtime = _required(raw, "runtime")
    decision = raw.get("decision", {})
    ratio = float(_required(model, "coreset_sampling_ratio"))
    if not 0 < ratio <= 1:
        raise ValueError("model.coreset_sampling_ratio must be in (0, 1]")
    size = tuple(int(value) for value in _required(data, "image_size"))
    if len(size) != 2 or min(size) <= 0:
        raise ValueError("data.image_size must contain two positive integers")
    threshold = decision.get("threshold")
    api = raw.get("api", {})
    if threshold is not None and not 0 <= float(threshold) <= 1:
        raise ValueError("decision.threshold must be in [0, 1]")

    def env(name: str, default: Any) -> Any:
        return os.getenv(f"VISIONGUARD_{name}", default)

    return AppConfig(
        name=str(_required(project, "name")),
        seed=int(project.get("seed", 42)),
        data=DataConfig(
            root=Path(_required(data, "root")),
            category=str(_required(data, "category")),
            image_size=(size[0], size[1]),
            train_batch_size=int(data.get("train_batch_size", 8)),
            eval_batch_size=int(data.get("eval_batch_size", 8)),
            num_workers=int(data.get("num_workers", 4)),
        ),
        model=ModelConfig(
            backbone=str(model.get("backbone", "wide_resnet50_2")),
            layers=tuple(str(layer) for layer in model.get("layers", ["layer2", "layer3"])),
            coreset_sampling_ratio=ratio,
            num_neighbors=int(model.get("num_neighbors", 9)),
        ),
        runtime=RuntimeConfig(
            accelerator=str(runtime.get("accelerator", "auto")),
            devices=int(runtime.get("devices", 1)),
            output_dir=Path(runtime.get("output_dir", "artifacts")),
            log_level=str(runtime.get("log_level", "INFO")),
        ),
        decision=DecisionConfig(threshold=None if threshold is None else float(threshold)),
        api=ApiConfig(
            host=str(env("API_HOST", api.get("host", "127.0.0.1"))),
            port=int(env("API_PORT", api.get("port", 8000))),
            allowed_origins=tuple(api.get("allowed_origins", ["http://localhost:3000"])),
            max_upload_bytes=int(env("MAX_UPLOAD_BYTES", api.get("max_upload_bytes", 10_000_000))),
            allowed_extensions=tuple(
                str(item).lower()
                for item in api.get("allowed_extensions", [".png", ".jpg", ".jpeg"])
            ),
            output_dir=Path(
                env("INSPECTION_OUTPUT_DIR", api.get("output_dir", "outputs/inspections"))
            ),
            checkpoint=(
                Path(value)
                if (value := env("CHECKPOINT", api.get("checkpoint")))
                else None
            ),
        ),
    )
