"""Tests for cv_lib.viz.batch — no GPU, no ultralytics required."""

from __future__ import annotations

import numpy as np
import pytest

from cv_lib.viz.batch import show_batch


def _img(h: int = 100, w: int = 100) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_show_batch_returns_ndarray():
    imgs = [_img() for _ in range(3)]
    grid = show_batch(imgs, show=False)
    assert isinstance(grid, np.ndarray)
    assert grid.dtype == np.uint8
    assert grid.ndim == 3


def test_grid_dimensions_no_labels():
    imgs = [_img(100, 100) for _ in range(6)]
    grid = show_batch(imgs, tile_size=(80, 80), cols=3, show=False)
    # 2 rows × 3 cols → 160 × 240
    assert grid.shape == (160, 240, 3)


def test_grid_pads_to_full_rows():
    imgs = [_img() for _ in range(5)]
    grid = show_batch(imgs, tile_size=(50, 50), cols=3, show=False)
    # ceil(5/3)=2 rows → 100 × 150
    assert grid.shape == (100, 150, 3)


def test_labels_drawn_without_crash():
    imgs = [_img(100, 100) for _ in range(2)]
    labels = [
        np.array([[0, 0.5, 0.5, 0.4, 0.3]]),
        np.array([[1, 0.2, 0.3, 0.1, 0.2]]),
    ]
    grid = show_batch(imgs, labels=labels, class_names=["cat", "dog"],
                      tile_size=(100, 100), cols=2, show=False)
    assert grid.shape[0] == 100
    assert grid.shape[1] == 200


def test_save_to_file(tmp_path):
    import cv2
    imgs = [_img() for _ in range(2)]
    out = tmp_path / "grid.png"
    show_batch(imgs, tile_size=(64, 64), cols=2, output_path=out, show=False)
    assert out.exists()
    loaded = cv2.imread(str(out))
    assert loaded is not None


def test_file_paths_as_input(tmp_path):
    import cv2
    p = tmp_path / "img.png"
    cv2.imwrite(str(p), _img(80, 80))
    grid = show_batch([str(p), str(p)], tile_size=(40, 40), cols=2, show=False)
    assert grid.shape == (40, 80, 3)
