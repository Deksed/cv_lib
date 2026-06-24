"""Tests for cv_lib.viz.errors — no GPU, no ultralytics required."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from cv_lib.viz.errors import find_errors, render_errors


@pytest.fixture()
def simple_dataset(tmp_path: Path):
    """
    2 images:
      img_a — GT has 1 box, prediction matches it  → no error
      img_b — GT has 1 box, no prediction          → 1 FN
               prediction has 1 extra box          → 1 FP
    """
    img_dir = tmp_path / "images"
    gt_dir = tmp_path / "labels"
    pred_dir = tmp_path / "pred_labels"
    for d in (img_dir, gt_dir, pred_dir):
        d.mkdir()

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(img_dir / "img_a.jpg"), img)
    cv2.imwrite(str(img_dir / "img_b.jpg"), img)

    # GT: img_a has one car box (centre of image)
    (gt_dir / "img_a.txt").write_text("0 0.5 0.5 0.4 0.4\n")
    # GT: img_b has one car box
    (gt_dir / "img_b.txt").write_text("0 0.5 0.5 0.4 0.4\n")

    # Predictions: img_a matches GT (same box, high conf)
    (pred_dir / "img_a.txt").write_text("0 0.5 0.5 0.4 0.4 0.95\n")
    # Predictions: img_b has a completely different box (FP), GT box missed (FN)
    (pred_dir / "img_b.txt").write_text("0 0.1 0.1 0.05 0.05 0.80\n")

    return img_dir, gt_dir, pred_dir


def test_find_errors_fp_fn(simple_dataset):
    img_dir, gt_dir, pred_dir = simple_dataset
    errors = find_errors(
        images_dir=img_dir,
        labels_dir=gt_dir,
        pred_labels_dir=pred_dir,
        conf_threshold=0.25,
        iou_threshold=0.5,
    )
    types = [e.error_type for e in errors]
    assert "FP" in types
    assert "FN" in types
    # img_a should be a clean match — no errors from it
    assert all(e.image_path.name != "img_a.jpg" for e in errors)


def test_find_errors_no_errors_on_perfect_match(simple_dataset):
    img_dir, gt_dir, pred_dir = simple_dataset
    errors = find_errors(
        images_dir=img_dir,
        labels_dir=gt_dir,
        pred_labels_dir=pred_dir,
    )
    assert all(e.image_path.name != "img_a.jpg" for e in errors)


def test_render_errors_returns_ndarray(simple_dataset):
    img_dir, gt_dir, pred_dir = simple_dataset
    errors = find_errors(img_dir, gt_dir, pred_labels_dir=pred_dir)
    grid = render_errors(errors, class_names=["car"], show=False)
    assert isinstance(grid, np.ndarray)
    assert grid.dtype == np.uint8
    assert grid.ndim == 3


def test_render_errors_empty_list():
    grid = render_errors([], show=False)
    assert isinstance(grid, np.ndarray)


def test_find_errors_no_pred_labels_all_fn(simple_dataset):
    img_dir, gt_dir, _ = simple_dataset
    empty_pred = img_dir.parent / "empty_pred"
    empty_pred.mkdir()
    errors = find_errors(img_dir, gt_dir, pred_labels_dir=empty_pred)
    assert all(e.error_type == "FN" for e in errors)
    assert len(errors) == 2


def test_find_errors_raises_without_source(simple_dataset):
    img_dir, gt_dir, _ = simple_dataset
    with pytest.raises(ValueError):
        find_errors(img_dir, gt_dir)
