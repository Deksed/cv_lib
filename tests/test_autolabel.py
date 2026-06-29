"""Tests for cv_lib.data.autolabel (model-driven pre-annotation)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from cv_lib.data.autolabel import autolabel


class _FakeBoxes:
    def __init__(self, xyxy, cls, conf):
        self.xyxy = np.array(xyxy, dtype=float)
        self.cls = np.array(cls, dtype=float)
        self.conf = np.array(conf, dtype=float)

    def __len__(self):
        return len(self.xyxy)


class _FakeModel:
    """Returns one detection per image at a fixed pixel box."""

    def predict(self, source, **kwargs):
        result = SimpleNamespace(
            boxes=_FakeBoxes([[20, 20, 60, 60]], [1.0], [0.9]),
            orig_shape=(100, 100),  # (h, w)
        )
        return [result]


def _make_images(d: Path, n: int = 2) -> None:
    d.mkdir(parents=True)
    for i in range(n):
        cv2.imwrite(str(d / f"img{i}.jpg"), np.full((100, 100, 3), 50, np.uint8))


def test_autolabel_writes_yolo(tmp_path: Path):
    images = tmp_path / "images"
    _make_images(images, 2)
    out = tmp_path / "labels"

    n = autolabel(_FakeModel(), images, out, conf=0.4)
    assert n == 2
    txt = (out / "img0.txt").read_text().strip().split()
    # class 1; box (20,20,60,60) on 100x100 → cx=0.4 cy=0.4 w=0.4 h=0.4
    assert txt[0] == "1"
    assert abs(float(txt[1]) - 0.4) < 1e-4
    assert abs(float(txt[3]) - 0.4) < 1e-4


def test_autolabel_save_conf(tmp_path: Path):
    images = tmp_path / "images"
    _make_images(images, 1)
    out = tmp_path / "labels"
    autolabel(_FakeModel(), images, out, save_conf=True)
    parts = (out / "img0.txt").read_text().strip().split()
    assert len(parts) == 6  # cls cx cy w h conf
    assert abs(float(parts[5]) - 0.9) < 1e-4
