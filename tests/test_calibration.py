from types import SimpleNamespace

import pytest

from visionguard.calibration import (
    LabeledScore,
    evaluate_threshold,
    prediction_scores,
    select_threshold,
)


def test_select_threshold_maximizes_balanced_accuracy() -> None:
    scores = [
        LabeledScore("good-a.png", 0, 0.10),
        LabeledScore("good-b.png", 0, 0.20),
        LabeledScore("bad-a.png", 1, 0.70),
        LabeledScore("bad-b.png", 1, 0.90),
    ]
    result = select_threshold(scores)
    assert result.balanced_accuracy == 1.0
    assert 0.20 < result.threshold <= 0.70
    assert result.false_positive == 0
    assert result.false_negative == 0


def test_calibration_requires_both_classes() -> None:
    with pytest.raises(ValueError, match="both normal and anomalous"):
        select_threshold([LabeledScore("good.png", 0, 0.1)])


def test_evaluate_threshold_returns_confusion_counts() -> None:
    scores = [
        LabeledScore("good.png", 0, 0.6),
        LabeledScore("bad.png", 1, 0.4),
    ]
    result = evaluate_threshold(scores, 0.5)
    assert (result.true_positive, result.false_positive) == (0, 1)
    assert (result.true_negative, result.false_negative) == (0, 1)


def test_prediction_scores_flattens_tensor_like_values() -> None:
    batch = SimpleNamespace(
        pred_score=[0.2, 0.8], image_path=["good.png", "bad.png"]
    )
    assert prediction_scores(batch, label=1) == [
        LabeledScore("good.png", 1, 0.2),
        LabeledScore("bad.png", 1, 0.8),
    ]
