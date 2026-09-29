from datetime import UTC, datetime, timedelta
from pathlib import Path

from visionguard.api.models import InspectionResult, ModelInfo
from visionguard.api.persistence import InspectionRepository


def result(identifier: str, decision: str, score: float) -> InspectionResult:
    return InspectionResult(
        inspection_id=identifier,
        filename=f"{identifier}.png",
        decision=decision,  # type: ignore[arg-type]
        anomaly_score=score,
        threshold=None,
        threshold_source="model_provisional",
        inference_time_ms=10.0,
        heatmap_url=f"/heatmaps/{identifier}",
        model=ModelInfo(category="sheet_metal", backbone="wide_resnet50_2"),
    )


def test_repository_history_filter_and_analytics(tmp_path: Path) -> None:
    repository = InspectionRepository(f"sqlite:///{tmp_path / 'test.db'}")
    repository.initialize()
    repository.save(result("pass-id", "PASS", 0.1))
    repository.save(result("fail-id", "FAIL", 0.9))

    items, total = repository.list(limit=10, offset=0)
    failed, failed_total = repository.list(limit=10, offset=0, decision="FAIL")
    summary = repository.analytics()

    assert total == 2
    assert {item.inspection_id for item in items} == {"pass-id", "fail-id"}
    assert failed_total == 1 and failed[0].inspection_id == "fail-id"
    assert repository.get("pass-id") is not None
    assert repository.get("missing") is None
    assert summary.total_inspections == 2
    assert summary.pass_count == 1
    assert summary.fail_count == 1
    assert summary.fail_rate == 0.5


def test_repository_retention(tmp_path: Path) -> None:
    repository = InspectionRepository(f"sqlite:///{tmp_path / 'test.db'}")
    repository.initialize()
    old = result("old-id", "PASS", 0.1).model_copy(
        update={"created_at": datetime.now(UTC) - timedelta(days=40)}
    )
    repository.save(old)
    assert repository.delete_expired(30) == 1
    assert repository.get("old-id") is None
