"""Tests for cv_lib.infer.tiled (tile geometry, NMS, sliced predict)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from cv_lib.infer.tiled import generate_tiles, nms_numpy, sliced_predict


def test_generate_tiles_covers_frame():
    tiles = generate_tiles(1000, 500, tile=400, overlap=0.25)
    # every tile fits inside the frame
    for x0, y0, x1, y1 in tiles:
        assert 0 <= x0 < x1 <= 1000
        assert 0 <= y0 < y1 <= 500
    # right and bottom edges are reached
    assert max(t[2] for t in tiles) == 1000
    assert max(t[3] for t in tiles) == 500


def test_generate_tiles_small_image_single_tile():
    tiles = generate_tiles(300, 200, tile=640, overlap=0.2)
    assert tiles == [(0, 0, 300, 200)]


def test_nms_suppresses_overlap():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=float)
    scores = np.array([0.9, 0.8, 0.7])
    keep = nms_numpy(boxes, scores, iou_threshold=0.5)
    assert 0 in keep  # highest score kept
    assert 1 not in keep  # heavy overlap with 0 suppressed
    assert 2 in keep  # disjoint box kept


def test_nms_empty():
    assert nms_numpy(np.zeros((0, 4)), np.zeros(0)) == []


def test_sliced_predict_offsets_and_merges():
    # Fake model: every tile reports one box at its local (5,5,15,15).
    class FakeBoxes:
        def __init__(self):
            self.xyxy = np.array([[5, 5, 15, 15]], dtype=float)
            self.cls = np.array([0.0])
            self.conf = np.array([0.9])

        def __len__(self):
            return 1

    class FakeModel:
        def predict(self, source, **kwargs):
            return [SimpleNamespace(boxes=FakeBoxes())]

    image = np.zeros((500, 500, 3), dtype=np.uint8)
    det = sliced_predict(FakeModel(), image, tile=250, overlap=0.0, nms_iou=0.5)
    # 2x2 tiles → 4 boxes, all disjoint after offset → 4 kept
    assert len(det["boxes"]) == 4
    # boxes must be shifted out of the first tile's local frame
    assert det["boxes"][:, 0].max() > 250
