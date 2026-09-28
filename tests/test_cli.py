from types import SimpleNamespace

from visionguard.cli import _summarize


def test_summarize_applies_manual_pass_fail_threshold() -> None:
    passing = _summarize(SimpleNamespace(pred_score=[0.2]), threshold=0.5)
    failing = _summarize(SimpleNamespace(pred_score=[0.8]), threshold=0.5)
    assert passing["decision"] == "PASS"
    assert failing["decision"] == "FAIL"
    assert passing["decision_source"] == "configured_threshold"


def test_summarize_uses_model_label_without_manual_threshold() -> None:
    result = _summarize(
        SimpleNamespace(pred_score=[0.7], pred_label=[True], image_path=["part.png"])
    )
    assert result == {
        "pred_score": 0.7,
        "pred_label": True,
        "image_path": "part.png",
        "decision": "FAIL",
        "decision_source": "model_threshold",
    }
