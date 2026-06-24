"""Tests for cv_lib.data.inspect — no GPU, no ultralytics required."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from cv_lib.data.inspect import inspect_dataset


@pytest.fixture()
def dataset(tmp_path: Path):
    img_dir = tmp_path / "images"
    lbl_dir = tmp_path / "labels"
    img_dir.mkdir(); lbl_dir.mkdir()

    # valid image + label
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(img_dir / "a.jpg"), img)
    (lbl_dir / "a.txt").write_text("0 0.5 0.5 0.4 0.3\n")

    # image with no label
    cv2.imwrite(str(img_dir / "b.jpg"), img)

    # image with invalid box (center out of range)
    cv2.imwrite(str(img_dir / "c.jpg"), img)
    (lbl_dir / "c.txt").write_text("0 1.5 0.5 0.2 0.2\n")

    # corrupt "image"
    (img_dir / "d.jpg").write_bytes(b"not an image")

    return img_dir, lbl_dir


def test_images_total(dataset):
    img_dir, lbl_dir = dataset
    report = inspect_dataset(img_dir, lbl_dir, num_classes=2)
    assert report.images_total == 4


def test_corrupt_detected(dataset):
    img_dir, lbl_dir = dataset
    report = inspect_dataset(img_dir, lbl_dir, num_classes=2)
    assert len(report.corrupt_images) == 1
    assert report.corrupt_images[0].name == "d.jpg"


def test_missing_label_detected(dataset):
    img_dir, lbl_dir = dataset
    report = inspect_dataset(img_dir, lbl_dir, num_classes=2)
    assert any(p.name == "b.jpg" for p in report.missing_labels)


def test_invalid_box_detected(dataset):
    img_dir, lbl_dir = dataset
    report = inspect_dataset(img_dir, lbl_dir, num_classes=2)
    assert len(report.invalid_boxes) >= 1
    assert any("center out of" in reason for _, _, reason in report.invalid_boxes)


def test_class_counts(dataset):
    img_dir, lbl_dir = dataset
    report = inspect_dataset(img_dir, lbl_dir, num_classes=2)
    # only a.txt is valid — 1 box for class 0
    assert report.class_counts is not None
    assert report.class_counts[0] == 1


def test_infer_labels_dir(tmp_path: Path):
    img_dir = tmp_path / "images" / "val"
    lbl_dir = tmp_path / "labels" / "val"
    img_dir.mkdir(parents=True); lbl_dir.mkdir(parents=True)
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    cv2.imwrite(str(img_dir / "x.png"), img)
    (lbl_dir / "x.txt").write_text("0 0.5 0.5 0.3 0.3\n")
    report = inspect_dataset(img_dir, num_classes=1)
    assert report.images_total == 1
    assert len(report.missing_labels) == 0
