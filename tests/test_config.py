from pathlib import Path

import pytest

from visionguard.config import load_config


def test_default_config_loads() -> None:
    config = load_config(Path("configs/patchcore_mvtecad2.yaml"))
    assert config.name == "visionguard-ai"
    assert config.data.image_size == (256, 256)
    assert config.model.coreset_sampling_ratio == 0.01


def test_invalid_coreset_ratio_is_rejected(tmp_path: Path) -> None:
    config_file = tmp_path / "bad.yaml"
    config_file.write_text(
        """
project: {name: test}
data: {root: data, category: test, image_size: [256, 256]}
model: {coreset_sampling_ratio: 0}
runtime: {}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="coreset_sampling_ratio"):
        load_config(config_file)
