"""Command-line interface for training, evaluation and single-image inspection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from visionguard.config import AppConfig, load_config
from visionguard.logging import configure_logging
from visionguard.pipeline import (
    build_components,
    find_checkpoint,
    load_trained_components,
    require_anomalib,
)
from visionguard.visualization import save_anomaly_overlay

app = typer.Typer(help="VisionGuard AI anomaly detection commands.", no_args_is_help=True)
ConfigOption = Annotated[Path, typer.Option("--config", "-c", exists=True, dir_okay=False)]


def _setup(config_path: Path) -> tuple[AppConfig, Any]:
    config = load_config(config_path)
    logger = configure_logging(config.runtime.log_level, config.runtime.output_dir)
    return config, logger


@app.command()
def train(config_path: ConfigOption = Path("configs/patchcore_mvtecad2.yaml")) -> None:
    """Fit PatchCore's memory bank and save a checkpoint."""
    config, logger = _setup(config_path)
    datamodule, model, engine = build_components(config)
    logger.info("Training category=%s", config.data.category)
    engine.fit(model=model, datamodule=datamodule)
    logger.info("Training complete; artifacts saved to %s", config.runtime.output_dir)


@app.command()
def evaluate(
    config_path: ConfigOption = Path("configs/patchcore_mvtecad2.yaml"),
    output: Annotated[Path, typer.Option("--output", "-o")] = Path(
        "artifacts/evaluation_metrics.json"
    ),
) -> None:
    """Evaluate the most recent checkpoint on the test split."""
    config, logger = _setup(config_path)
    datamodule, model, engine = build_components(config)
    checkpoint = find_checkpoint(config.runtime.output_dir)
    logger.info("Evaluating checkpoint=%s", checkpoint)
    metrics = engine.test(model=model, datamodule=datamodule, ckpt_path=str(checkpoint))
    serialized = json.dumps(metrics, indent=2, default=str)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialized, encoding="utf-8")
    logger.info("Evaluation metrics saved to %s", output)
    print(serialized)


@app.command("calibrate-threshold")
def calibrate_threshold(
    good_dir: Annotated[Path, typer.Option("--good-dir", exists=True, file_okay=False)] = Path(
        "data/mvtecad2/sheet_metal/test_public/good"
    ),
    bad_dir: Annotated[Path, typer.Option("--bad-dir", exists=True, file_okay=False)] = Path(
        "data/mvtecad2/sheet_metal/test_public/bad"
    ),
    pattern: Annotated[str, typer.Option("--pattern")] = "*_regular.png",
    good_pattern: Annotated[str | None, typer.Option("--good-pattern")] = None,
    bad_pattern: Annotated[str | None, typer.Option("--bad-pattern")] = None,
    scope: Annotated[str, typer.Option("--scope")] = "provisional_labeled_calibration",
    config_path: ConfigOption = Path("configs/patchcore_mvtecad2.yaml"),
    output: Annotated[Path, typer.Option("--output", "-o")] = Path(
        "artifacts/threshold_calibration.json"
    ),
) -> None:
    """Calibrate a provisional threshold from explicitly labeled image folders."""
    from visionguard.calibration import score_images, select_threshold, serialize_metrics

    config, logger = _setup(config_path)
    good_paths = sorted(good_dir.glob(good_pattern or pattern))
    bad_paths = sorted(bad_dir.glob(bad_pattern or pattern))
    if not good_paths or not bad_paths:
        raise typer.BadParameter("Both folders must contain files matching --pattern")
    checkpoint, model, engine = load_trained_components(config)
    logger.info(
        "Calibrating threshold checkpoint=%s normal=%d anomalous=%d",
        checkpoint,
        len(good_paths),
        len(bad_paths),
    )
    scores = score_images(engine, model, good_paths, label=0)
    scores.extend(score_images(engine, model, bad_paths, label=1))
    selected = select_threshold(scores)
    payload = {
        "method": "maximum_balanced_accuracy",
        "scope": scope,
        "pattern": pattern,
        "good_pattern": good_pattern or pattern,
        "bad_pattern": bad_pattern or pattern,
        "normal_samples": len(good_paths),
        "anomalous_samples": len(bad_paths),
        "checkpoint": str(checkpoint),
        "metrics": serialize_metrics(selected),
        "scores": [item.__dict__ for item in scores],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["metrics"], indent=2))
    print(f"Calibration result saved to {output}")


@app.command("evaluate-threshold")
def evaluate_decision_threshold(
    threshold: Annotated[float, typer.Option("--threshold", min=0.0, max=1.0)],
    good_dir: Annotated[Path, typer.Option("--good-dir", exists=True, file_okay=False)] = Path(
        "data/mvtecad2/sheet_metal/test_public/good"
    ),
    bad_dir: Annotated[Path, typer.Option("--bad-dir", exists=True, file_okay=False)] = Path(
        "data/mvtecad2/sheet_metal/test_public/bad"
    ),
    pattern: Annotated[str, typer.Option("--pattern")] = "*.png",
    exclude_pattern: Annotated[str | None, typer.Option("--exclude-pattern")] = None,
    config_path: ConfigOption = Path("configs/patchcore_mvtecad2.yaml"),
    output: Annotated[Path, typer.Option("--output", "-o")] = Path(
        "artifacts/threshold_evaluation.json"
    ),
) -> None:
    """Evaluate one fixed threshold without selecting it from the evaluated images."""
    from visionguard.calibration import evaluate_threshold, score_images, serialize_metrics

    config, logger = _setup(config_path)

    def selected_paths(directory: Path) -> list[Path]:
        excluded = set(directory.glob(exclude_pattern)) if exclude_pattern else set()
        return [path for path in sorted(directory.glob(pattern)) if path not in excluded]

    good_paths = selected_paths(good_dir)
    bad_paths = selected_paths(bad_dir)
    if not good_paths or not bad_paths:
        raise typer.BadParameter("Both folders must contain selected evaluation images")
    checkpoint, model, engine = load_trained_components(config)
    logger.info(
        "Evaluating threshold=%f checkpoint=%s normal=%d anomalous=%d",
        threshold,
        checkpoint,
        len(good_paths),
        len(bad_paths),
    )
    scores = score_images(engine, model, good_paths, label=0)
    scores.extend(score_images(engine, model, bad_paths, label=1))
    metrics = evaluate_threshold(scores, threshold)
    payload = {
        "method": "fixed_threshold_evaluation",
        "pattern": pattern,
        "exclude_pattern": exclude_pattern,
        "normal_samples": len(good_paths),
        "anomalous_samples": len(bad_paths),
        "checkpoint": str(checkpoint),
        "metrics": serialize_metrics(metrics),
        "scores": [item.__dict__ for item in scores],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["metrics"], indent=2))
    print(f"Threshold evaluation saved to {output}")


@app.command("api")
def serve_api(config_path: ConfigOption = Path("configs/patchcore_mvtecad2.yaml")) -> None:
    """Run the VisionGuard HTTP inference service."""
    import os

    import uvicorn

    config = load_config(config_path)
    os.environ["VISIONGUARD_CONFIG"] = str(config_path.resolve())
    uvicorn.run(
        "visionguard.api.app:create_app",
        factory=True,
        host=config.api.host,
        port=config.api.port,
        log_level=config.runtime.log_level.lower(),
    )


@app.command("db-backup")
def database_backup(
    config_path: ConfigOption = Path("configs/patchcore_mvtecad2.yaml"),
    output: Annotated[Path, typer.Option("--output", "-o")] = Path(
        "backups/inspections.json"
    ),
) -> None:
    """Export inspection metadata to a portable JSON backup."""
    from visionguard.api.persistence import InspectionRepository

    config = load_config(config_path)
    repository = InspectionRepository(config.database.url)
    repository.initialize()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = [json.loads(item.model_dump_json()) for item in repository.export_all()]
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Backed up {len(payload)} inspection records to {output}")


@app.command("db-restore")
def database_restore(
    backup: Annotated[Path, typer.Option("--backup", exists=True, dir_okay=False)],
    config_path: ConfigOption = Path("configs/patchcore_mvtecad2.yaml"),
) -> None:
    """Merge inspection metadata from a VisionGuard JSON backup."""
    from visionguard.api.models import InspectionResult
    from visionguard.api.persistence import InspectionRepository

    raw = json.loads(backup.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise typer.BadParameter("Backup must contain a JSON list")
    config = load_config(config_path)
    repository = InspectionRepository(config.database.url)
    repository.initialize()
    for item in raw:
        repository.save(InspectionResult.model_validate(item))
    print(f"Restored {len(raw)} inspection records from {backup}")


@app.command()
def predict(
    image: Annotated[Path, typer.Option("--image", "-i", exists=True, dir_okay=False)],
    config_path: ConfigOption = Path("configs/patchcore_mvtecad2.yaml"),
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("outputs/prediction"),
) -> None:
    """Inspect one image using the most recent trained checkpoint."""
    config, logger = _setup(config_path)
    _, engine_cls, patchcore_cls = require_anomalib()
    checkpoint = find_checkpoint(config.runtime.output_dir)
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
    output.mkdir(parents=True, exist_ok=True)
    logger.info("Inspecting image=%s", image)
    predictions = engine.predict(
        model=model,
        data_path=image,
        ckpt_path=str(checkpoint),
        return_predictions=True,
    )
    batch = predictions[0] if isinstance(predictions, list) and predictions else predictions
    anomaly_map = getattr(batch, "anomaly_map", None)
    if anomaly_map is not None:
        overlay_file = save_anomaly_overlay(image, anomaly_map, output / "heatmap_overlay.png")
        logger.info("Heatmap overlay saved to %s", overlay_file)
    result_file = output / "result.json"
    result_file.write_text(
        json.dumps(_summarize(predictions, config.decision.threshold), indent=2),
        encoding="utf-8",
    )
    print(f"Inspection result saved to {result_file}")


def _summarize(predictions: Any, threshold: float | None = None) -> dict[str, Any]:
    """Convert common anomalib prediction fields into JSON-safe output."""
    batch = predictions[0] if isinstance(predictions, list) and predictions else predictions
    summary: dict[str, Any] = {}
    for name in ("pred_score", "pred_label", "image_path"):
        value = getattr(batch, name, None)
        if value is not None:
            if hasattr(value, "detach"):
                value = value.detach().cpu().tolist()
            if isinstance(value, (list, tuple)):
                value = [str(item) if isinstance(item, Path) else item for item in value]
                if len(value) == 1:
                    value = value[0]
            summary[name] = str(value) if isinstance(value, Path) else value
    score = summary.get("pred_score")
    if threshold is not None and isinstance(score, (int, float)):
        summary["threshold"] = threshold
        summary["decision"] = "FAIL" if score >= threshold else "PASS"
        summary["decision_source"] = "configured_threshold"
    elif isinstance(summary.get("pred_label"), bool):
        summary["decision"] = "FAIL" if summary["pred_label"] else "PASS"
        summary["decision_source"] = "model_threshold"
    return summary or {"status": "prediction_completed"}


if __name__ == "__main__":
    app()
