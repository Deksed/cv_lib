"""Batch visualization: show a grid of images with YOLO bounding box overlays."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from cv_lib.viz._utils import assemble_grid, display, draw_boxes_on, to_bgr_uint8

if TYPE_CHECKING:
    import torch


def show_batch(
    images: list[np.ndarray | torch.Tensor | str | Path],
    labels: list[np.ndarray] | None = None,
    class_names: list[str] | None = None,
    tile_size: tuple[int, int] = (320, 320),
    cols: int = 4,
    output_path: str | Path | None = None,
    show: bool = True,
) -> np.ndarray:
    """
    Display or save a grid of images, optionally with YOLO bbox overlays.

    Args:
        images:       list of images — ndarray (H×W×C uint8 BGR), Tensor (C×H×W float),
                      or file path strings/Paths.
        labels:       optional list of per-image label arrays, shape (N, 5) with columns
                      [class_id, cx, cy, w, h] in YOLO normalised format.
        class_names:  class name list for box labels.
        tile_size:    (width, height) to which each tile is resized.
        cols:         number of columns in the grid.
        output_path:  if given, saves the grid to this path.
        show:         display with matplotlib (Jupyter) or cv2.imshow (terminal).

    Returns:
        Grid as H×W×C uint8 BGR ndarray.
    """
    if cols < 1:
        raise ValueError(f"cols must be >= 1, got {cols}")
    class_names = class_names or []
    tw, th = tile_size

    tiles: list[np.ndarray] = []
    for idx, img in enumerate(images):
        if isinstance(img, (str, Path)):
            bgr = cv2.imread(str(img))
            if bgr is None:
                bgr = np.zeros((th, tw, 3), dtype=np.uint8)
        else:
            bgr = to_bgr_uint8(img)

        if labels is not None and idx < len(labels) and labels[idx] is not None:
            lbl = np.asarray(labels[idx])
            if lbl.ndim == 2 and lbl.shape[1] >= 5:
                class_ids = lbl[:, 0].astype(int)
                boxes_yolo = lbl[:, 1:5]
                bgr = draw_boxes_on(bgr, boxes_yolo, class_ids, class_names)

        tiles.append(cv2.resize(bgr, (tw, th)))

    grid = assemble_grid(tiles, cols, tile_size)

    if output_path is not None:
        cv2.imwrite(str(output_path), grid)

    if show:
        display(grid, window="batch")

    return grid
