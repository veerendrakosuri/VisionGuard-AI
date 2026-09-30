"""Threshold calibration and evaluation for labeled image-level anomaly scores."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LabeledScore:
    path: str
    label: int
    score: float


@dataclass(frozen=True)
class ThresholdMetrics:
    threshold: float
    balanced_accuracy: float
    sensitivity: float
    specificity: float
    precision: float
    f1: float
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int


def evaluate_threshold(scores: Sequence[LabeledScore], threshold: float) -> ThresholdMetrics:
    """Measure an image-level FAIL-if-score-at-least-threshold rule."""
    if not scores:
        raise ValueError("At least one labeled score is required")
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be in [0, 1]")
    tp = sum(item.label == 1 and item.score >= threshold for item in scores)
    fp = sum(item.label == 0 and item.score >= threshold for item in scores)
    tn = sum(item.label == 0 and item.score < threshold for item in scores)
    fn = sum(item.label == 1 and item.score < threshold for item in scores)
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else 0.0
    return ThresholdMetrics(
        threshold=threshold,
        balanced_accuracy=(sensitivity + specificity) / 2,
        sensitivity=sensitivity,
        specificity=specificity,
        precision=precision,
        f1=f1,
        true_positive=tp,
        false_positive=fp,
        true_negative=tn,
        false_negative=fn,
    )


def select_threshold(scores: Sequence[LabeledScore]) -> ThresholdMetrics:
    """Choose the threshold with best balanced accuracy, then sensitivity and F1."""
    labels = {item.label for item in scores}
    if labels != {0, 1}:
        raise ValueError("Calibration requires both normal and anomalous samples")
    for item in scores:
        if not 0 <= item.score <= 1:
            raise ValueError("Calibration scores must be in [0, 1]")
    unique_scores = sorted({item.score for item in scores})
    candidates = [0.0, 1.0]
    candidates.extend(unique_scores)
    candidates.extend((left + right) / 2 for left, right in pairwise(unique_scores))
    results = [evaluate_threshold(scores, candidate) for candidate in sorted(set(candidates))]
    return max(
        results,
        key=lambda result: (
            result.balanced_accuracy,
            result.sensitivity,
            result.f1,
            result.specificity,
            result.threshold,
        ),
    )


def prediction_scores(predictions: Any, label: int) -> list[LabeledScore]:
    """Extract per-image paths and normalized scores from Anomalib predictions."""
    batches = predictions if isinstance(predictions, list) else [predictions]
    extracted: list[LabeledScore] = []
    for batch in batches:
        raw_scores = _as_list(getattr(batch, "pred_score", None))
        raw_paths = _as_list(getattr(batch, "image_path", None))
        if len(raw_scores) != len(raw_paths):
            raise ValueError("Prediction score and image path counts do not match")
        for raw_path, raw_score in zip(raw_paths, raw_scores, strict=True):
            score = float(raw_score)
            extracted.append(LabeledScore(path=str(raw_path), label=label, score=score))
    return extracted


def score_images(engine: Any, model: Any, paths: Iterable[Path], label: int) -> list[LabeledScore]:
    """Run one loaded model over selected paths without modifying source images."""
    results: list[LabeledScore] = []
    for path in paths:
        predictions = engine.predict(model=model, data_path=path, return_predictions=True)
        results.extend(prediction_scores(predictions, label))
    return results


def serialize_metrics(metrics: ThresholdMetrics) -> dict[str, float | int]:
    return asdict(metrics)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]
