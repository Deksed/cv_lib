"""Tests for cv_lib.viz.compare — YOLO label parsing, label-path resolution,
non-mutating box drawing, and the GT-vs-pred side-by-side (model stubbed)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from cv_lib.viz.compare import (
    _draw_boxes,
    _resolve_label_path,
    compare_gt_pred,
    load_yolo_gt,
)


def test_load_yolo_gt_converts_to_pixel_xyxy(tmp_path: Path):
    label = tmp_path / "a.txt"
    # one box: class 1, centre (0.5,0.5), w=0.2 h=0.4 in a 100x200 image
    label.write_text("1 0.5 0.5 0.2 0.4\n")
    boxes, classes = load_yolo_gt(label, img_w=100, img_h=200)

    assert boxes.shape == (1, 4)
    assert classes.tolist() == [1]
    # x: (0.5±0.1)*100 = 40,60 ; y: (0.5±0.2)*200 = 60,140
    np.testing.assert_allclose(boxes[0], [40, 60, 60, 140], atol=1e-3)


def test_load_yolo_gt_missing_file_is_empty(tmp_path: Path):
    boxes, classes = load_yolo_gt(tmp_path / "nope.txt", 100, 100)
    assert boxes.shape == (0, 4)
    assert classes.shape == (0,)


def test_load_yolo_gt_blank_lines_ignored(tmp_path: Path):
    label = tmp_path / "a.txt"
    label.write_text("\n  \n")
    boxes, _ = load_yolo_gt(label, 100, 100)
    assert boxes.shape == (0, 4)


def test_resolve_label_path_same_dir(tmp_path: Path):
    img = tmp_path / "frame.jpg"
    img.write_bytes(b"x")
    (tmp_path / "frame.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    assert _resolve_label_path(img) == tmp_path / "frame.txt"


def test_resolve_label_path_images_labels_swap(tmp_path: Path):
    img_dir = tmp_path / "images" / "val"
    lbl_dir = tmp_path / "labels" / "val"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)
    img = img_dir / "frame.jpg"
    img.write_bytes(b"x")
    (lbl_dir / "frame.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    assert _resolve_label_path(img) == lbl_dir / "frame.txt"


def test_draw_boxes_does_not_mutate_input():
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    before = img.copy()
    boxes = np.array([[5, 5, 40, 40]], dtype=np.float32)
    out = _draw_boxes(img, boxes, np.array([0]), ["car"])

    assert np.array_equal(img, before)  # input untouched (drawn on a copy)
    assert not np.array_equal(out, before)  # something was drawn
    assert out.shape == img.shape


class _FakeBoxes:
    def __len__(self):
        return 0


class _FakeResult:
    boxes = _FakeBoxes()


class _FakeYOLO:
    def __init__(self, *a, **k):
        pass

    def __call__(self, *a, **k):
        return [_FakeResult()]


def test_compare_gt_pred_side_by_side_shape(tmp_path: Path, monkeypatch):
    # real image on disk + a GT label; predictions stubbed to empty
    img = np.zeros((60, 80, 3), dtype=np.uint8)
    img_path = tmp_path / "frame.jpg"
    cv2.imwrite(str(img_path), img)
    (tmp_path / "frame.txt").write_text("0 0.5 0.5 0.4 0.4\n")

    import ultralytics

    monkeypatch.setattr(ultralytics, "YOLO", _FakeYOLO)

    out_path = tmp_path / "cmp.png"
    result = compare_gt_pred(
        img_path, "model.pt", class_names=["car"], output_path=out_path, show=False
    )

    h, w = img.shape[:2]
    # two panels (w each) + a 4px divider
    assert result.shape == (h, w * 2 + 4, 3)
    assert out_path.exists()
