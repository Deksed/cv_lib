"""Tests for cv_lib.data.crops (object crop extraction)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from cv_lib.data.crops import extract_crops


def _setup(tmp_path: Path) -> tuple[Path, Path]:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    img = np.full((100, 100, 3), 128, np.uint8)
    cv2.imwrite(str(images / "frame.jpg"), img)
    # two boxes: class 0 (centre) and class 1 (corner)
    (labels / "frame.txt").write_text("0 0.5 0.5 0.4 0.4\n1 0.2 0.2 0.2 0.2\n")
    return images, labels


def test_extract_per_class(tmp_path: Path):
    images, labels = _setup(tmp_path)
    out = tmp_path / "crops"

    report = extract_crops(images, labels, out, per_class=True, class_names=["car", "person"])
    assert report.crops == 2
    assert report.images == 1
    assert (out / "car").is_dir()
    assert (out / "person").is_dir()
    assert report.per_class == {"car": 1, "person": 1}
    # crop dimensions roughly match the box (0.4*100 = 40px)
    crop = cv2.imread(str(next((out / "car").glob("*.jpg"))))
    assert abs(crop.shape[0] - 40) <= 2


def test_extract_flat(tmp_path: Path):
    images, labels = _setup(tmp_path)
    out = tmp_path / "crops"
    report = extract_crops(images, labels, out, per_class=False)
    assert report.crops == 2
    assert len(list(out.glob("*.jpg"))) == 2


def test_pad_enlarges_crop(tmp_path: Path):
    images, labels = _setup(tmp_path)
    base = extract_crops(images, labels, tmp_path / "c0", pad=0.0, per_class=False)
    padded = extract_crops(images, labels, tmp_path / "c1", pad=0.5, per_class=False)
    assert base.crops == padded.crops == 2
    c0 = cv2.imread(str(next((tmp_path / "c0").glob("frame_0*.jpg"))))
    c1 = cv2.imread(str(next((tmp_path / "c1").glob("frame_0*.jpg"))))
    assert c1.shape[0] >= c0.shape[0]
