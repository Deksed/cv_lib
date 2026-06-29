"""Tests for cv_lib.data.mining (hard-example ranking)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from cv_lib.data.mining import rank_for_labeling, uncertainty_score


def test_uncertainty_score_near_half_is_high():
    # confidences near 0.5 → uncertain → high score
    assert uncertainty_score([0.5, 0.5]) > uncertainty_score([0.95, 0.98])


def test_uncertainty_empty_is_max():
    assert uncertainty_score([]) == 1.0


def test_low_conf_strategy():
    assert uncertainty_score([0.1, 0.2], by="low_conf") > uncertainty_score([0.9], by="low_conf")


def test_num_detections_strategy():
    assert uncertainty_score([0.9, 0.9, 0.9], by="num_detections") == 3.0


class _FakeBoxes:
    def __init__(self, conf):
        self.conf = np.array(conf, dtype=float)

    def __len__(self):
        return len(self.conf)


class _FakeModel:
    """Maps filename → confidence list so ranking is deterministic."""

    def __init__(self, mapping):
        self.mapping = mapping

    def predict(self, source, **kwargs):
        confs = self.mapping[Path(source).name]
        boxes = _FakeBoxes(confs) if confs else None
        return [SimpleNamespace(boxes=boxes)]


def test_rank_orders_uncertain_first(tmp_path: Path):
    images = tmp_path / "images"
    images.mkdir()
    for name in ("sure.jpg", "unsure.jpg"):
        cv2.imwrite(str(images / name), np.full((40, 40, 3), 30, np.uint8))

    model = _FakeModel({"sure.jpg": [0.97], "unsure.jpg": [0.5]})
    ranked = rank_for_labeling(model, images)
    assert ranked[0][0].name == "unsure.jpg"  # most uncertain first
    assert ranked[0][1] > ranked[1][1]
