from pathlib import Path

import pytest

from visionguard.pipeline import find_checkpoint


def test_find_checkpoint_selects_latest(tmp_path: Path) -> None:
    older = tmp_path / "older.ckpt"
    newer = tmp_path / "nested" / "newer.ckpt"
    newer.parent.mkdir()
    older.write_text("old", encoding="utf-8")
    newer.write_text("new", encoding="utf-8")
    older.touch()
    newer.touch()
    assert find_checkpoint(tmp_path) == newer


def test_find_checkpoint_explains_missing_model(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Run the training command first"):
        find_checkpoint(tmp_path)
