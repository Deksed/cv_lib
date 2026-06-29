"""Inference helpers (tiled / sliced prediction for large frames)."""

from __future__ import annotations

from cv_lib.infer.tiled import generate_tiles, nms_numpy, sliced_predict

__all__ = ["generate_tiles", "nms_numpy", "sliced_predict"]
