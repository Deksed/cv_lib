"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture()
def sample_image() -> np.ndarray:
    """640×480 BGR uint8 image filled with a gradient."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:, :, 1] = np.linspace(0, 255, 640, dtype=np.uint8)  # green gradient
    return img


@pytest.fixture()
def yolo_label_file(tmp_path: Path) -> Path:
    """Single YOLO .txt label: one box per class 0 and class 1."""
    label = tmp_path / "sample.txt"
    label.write_text("0 0.5 0.5 0.4 0.3\n1 0.2 0.3 0.1 0.2\n")
    return label
