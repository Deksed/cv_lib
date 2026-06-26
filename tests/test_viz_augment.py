"""Tests for cv_lib.viz.augment — needs albumentations (a core dep), no GPU."""

from __future__ import annotations

import numpy as np
import pytest

from cv_lib.viz.augment import augment_preview, default_transform


def _img(h: int = 100, w: int = 100) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_returns_bgr_uint8_grid():
    grid = augment_preview(_img(), n=3, show=False)
    assert isinstance(grid, np.ndarray)
    assert grid.dtype == np.uint8
    assert grid.ndim == 3 and grid.shape[2] == 3


def test_grid_dimensions_include_original():
    # 1 original + 3 augs = 4 tiles, cols=2 → 2 rows × 2 cols.
    grid = augment_preview(_img(), n=3, tile_size=(64, 64), cols=2, show=False)
    assert grid.shape == (128, 128, 3)


def test_grid_pads_to_full_rows():
    # 1 + 3 = 4 tiles, cols=3 → ceil(4/3)=2 rows × 3 cols.
    grid = augment_preview(_img(), n=3, tile_size=(50, 50), cols=3, show=False)
    assert grid.shape == (100, 150, 3)


def test_with_boxes_does_not_crash():
    boxes = np.array([[0, 0.5, 0.5, 0.4, 0.3], [1, 0.2, 0.3, 0.1, 0.2]])
    grid = augment_preview(
        _img(120, 120), boxes, class_names=["cat", "dog"],
        n=4, tile_size=(80, 80), cols=2, show=False,
    )
    assert grid.shape[2] == 3


def test_deterministic_with_seed():
    boxes = np.array([[0, 0.5, 0.5, 0.4, 0.3]])
    a = augment_preview(_img(), boxes, n=4, seed=7, tile_size=(64, 64), show=False)
    b = augment_preview(_img(), boxes, n=4, seed=7, tile_size=(64, 64), show=False)
    assert np.array_equal(a, b)


def test_save_to_file(tmp_path):
    import cv2

    out = tmp_path / "aug.png"
    augment_preview(_img(), n=3, tile_size=(64, 64), output_path=out, show=False)
    assert out.exists()
    assert cv2.imread(str(out)) is not None


def test_file_path_as_input(tmp_path):
    import cv2

    p = tmp_path / "img.png"
    cv2.imwrite(str(p), _img(80, 80))
    grid = augment_preview(str(p), n=2, tile_size=(40, 40), cols=3, show=False)
    assert grid.shape[2] == 3


def test_missing_image_raises():
    with pytest.raises(FileNotFoundError):
        augment_preview("does_not_exist.png", show=False)


def test_default_transform_carries_bbox_params():
    t = default_transform()
    assert t.processors.get("bboxes") is not None
