"""Anomalib PatchCore pipeline construction and result normalization."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from visionguard.config import AppConfig


def require_anomalib() -> tuple[Any, Any, Any]:
    """Import heavyweight runtime dependencies with an actionable error."""
    cache_root = Path("work").resolve()
    cache_directories = {
        "MPLCONFIGDIR": cache_root / "matplotlib",
        "HF_HOME": cache_root / "huggingface",
        "TORCH_HOME": cache_root / "torch",
    }
    for environment_variable, cache_directory in cache_directories.items():
        cache_directory.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(environment_variable, str(cache_directory))
    try:
        from anomalib.data import MVTecAD2
        from anomalib.engine import Engine
        from anomalib.models import Patchcore
    except ImportError as exc:
        raise RuntimeError(
            "Anomalib is not installed. Run: python -m pip install -e ."
        ) from exc
    return MVTecAD2, Engine, Patchcore


def build_components(config: AppConfig) -> tuple[Any, Any, Any]:
    """Build the configured MVTec AD 2 datamodule, PatchCore model and engine."""
    mvtec_ad2, engine_cls, patchcore_cls = require_anomalib()
    datamodule = mvtec_ad2(
        root=config.data.root,
        category=config.data.category,
        train_batch_size=config.data.train_batch_size,
        eval_batch_size=config.data.eval_batch_size,
        num_workers=config.data.num_workers,
    )
    model = patchcore_cls(
        backbone=config.model.backbone,
        layers=list(config.model.layers),
        coreset_sampling_ratio=config.model.coreset_sampling_ratio,
        num_neighbors=config.model.num_neighbors,
    )
    engine = engine_cls(
        accelerator=config.runtime.accelerator,
        devices=config.runtime.devices,
        default_root_dir=config.runtime.output_dir,
    )
    return datamodule, model, engine


def load_trained_components(config: AppConfig) -> tuple[Path, Any, Any]:
    """Load the configured PatchCore checkpoint and create its inference engine."""
    checkpoint = config.api.checkpoint or find_checkpoint(config.runtime.output_dir)
    _, engine_cls, patchcore_cls = require_anomalib()
    import anomalib
    from torch.serialization import safe_globals

    # PyTorch 2.6+ only loads explicitly allowlisted non-tensor checkpoint globals.
    # This checkpoint is generated locally by VisionGuard's training command.
    with safe_globals([anomalib.PrecisionType]):
        model = patchcore_cls.load_from_checkpoint(str(checkpoint))
    engine = engine_cls(
        accelerator=config.runtime.accelerator,
        devices=config.runtime.devices,
        default_root_dir=config.runtime.output_dir,
    )
    return checkpoint, model, engine


def find_checkpoint(output_dir: Path) -> Path:
    checkpoints = sorted(output_dir.rglob("*.ckpt"), key=lambda item: item.stat().st_mtime)
    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoint found under {output_dir}. Run the training command first."
        )
    return checkpoints[-1]
