"""OpenCV rendering for anomaly localization maps."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import numpy.typing as npt


def _to_numpy(value: Any) -> npt.NDArray[np.float32]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float32).squeeze()
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D anomaly map after squeeze, got shape {array.shape}")
    return cast(npt.NDArray[np.float32], array)


def normalize_map(anomaly_map: Any) -> npt.NDArray[np.uint8]:
    """Scale an anomaly map to uint8 without dividing by zero."""
    array = _to_numpy(anomaly_map)
    minimum = float(array.min())
    maximum = float(array.max())
    if maximum <= minimum:
        return cast(npt.NDArray[np.uint8], np.zeros(array.shape, dtype=np.uint8))
    normalized = ((array - minimum) * (255.0 / (maximum - minimum))).astype(np.uint8)
    return cast(npt.NDArray[np.uint8], normalized)


def save_anomaly_overlay(image_path: Path, anomaly_map: Any, output_path: Path) -> Path:
    """Blend a colored anomaly map over the original image and save it."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not read image: {image_path}")
    normalized = normalize_map(anomaly_map)
    resized = cv2.resize(normalized, (image.shape[1], image.shape[0]))
    heatmap = cv2.applyColorMap(resized, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), overlay):
        raise OSError(f"Could not write heatmap overlay: {output_path}")
    return output_path
