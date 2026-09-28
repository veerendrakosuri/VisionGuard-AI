from types import SimpleNamespace

from visionguard.cli import _summarize


def test_summarize_applies_manual_pass_fail_threshold() -> None:
    passing = _summarize(SimpleNamespace(pred_score=[0.2]), threshold=0.5)
    failing = _summarize(SimpleNamespace(pred_score=[0.8]), threshold=0.5)
    assert passing["decision"] == "PASS"
    assert failing["decision"] == "FAIL"
