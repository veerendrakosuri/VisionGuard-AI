import numpy as np

from visionguard.visualization import normalize_map


def test_normalize_map_spans_uint8_range() -> None:
    result = normalize_map(np.array([[1.0, 2.0], [3.0, 5.0]], dtype=np.float32))
    assert result.dtype == np.uint8
    assert result.min() == 0
    assert result.max() == 255


def test_normalize_constant_map_is_zero() -> None:
    result = normalize_map(np.ones((2, 2), dtype=np.float32))
    assert not result.any()
