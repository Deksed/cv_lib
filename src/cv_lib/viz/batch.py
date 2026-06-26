"""Batch visualization: show a grid of images with YOLO bounding box overlays."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    import torch


def _to_bgr_uint8(img: np.ndarray | torch.Tensor) -> np.ndarray:
    """Accept H×W×C uint8 BGR ndarray or C×H×W float Tensor, return H×W×C uint8 BGR."""
    try:
        import torch
        if isinstance(img, torch.Tensor):
            t = img.detach().cpu()
            if t.ndim == 3:
                t = t.permute(1, 2, 0)  # C×H×W → H×W×C
            arr = t.numpy()
            if arr.dtype != np.uint8:
                arr = (arr * 255).clip(0, 255).astype(np.uint8)
            # assume RGB from torch → convert to BGR
            if arr.shape[2] == 3:
                arr = arr[:, :, ::-1].copy()
            return arr
    except ImportError:
        pass
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
    return arr


def _draw_boxes_on(
    img_bgr: np.ndarray,
    boxes_yolo: np.ndarray,
    class_ids: np.ndarray,
    class_names: list[str],
    color: tuple[int, int, int] = (0, 200, 0),
) -> np.ndarray:
    """Draw YOLO-format boxes (cx cy w h, normalised) onto a BGR copy of the image."""
    out = img_bgr.copy()
    h, w = out.shape[:2]
    for (cx, cy, bw, bh), cid in zip(boxes_yolo, class_ids):
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = class_names[int(cid)] if int(cid) < len(class_names) else str(int(cid))
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(out, (x1, y1 - th - base - 4), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - base - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return out


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
    class_names = class_names or []
    tw, th = tile_size

    tiles: list[np.ndarray] = []
    for idx, img in enumerate(images):
        if isinstance(img, (str, Path)):
            bgr = cv2.imread(str(img))
            if bgr is None:
                bgr = np.zeros((th, tw, 3), dtype=np.uint8)
        else:
            bgr = _to_bgr_uint8(img)

        if labels is not None and idx < len(labels) and labels[idx] is not None:
            lbl = np.asarray(labels[idx])
            if lbl.ndim == 2 and lbl.shape[1] >= 5:
                class_ids = lbl[:, 0].astype(int)
                boxes_yolo = lbl[:, 1:5]
                bgr = _draw_boxes_on(bgr, boxes_yolo, class_ids, class_names)

        tiles.append(cv2.resize(bgr, (tw, th)))

    # pad to full grid
    rows = (len(tiles) + cols - 1) // cols
    while len(tiles) < rows * cols:
        tiles.append(np.zeros((th, tw, 3), dtype=np.uint8))

    grid_rows = [np.hstack(tiles[r * cols: (r + 1) * cols]) for r in range(rows)]
    grid = np.vstack(grid_rows)

    if output_path is not None:
        cv2.imwrite(str(output_path), grid)

    if show:
        _display(grid)

    return grid


def _display(grid_bgr: np.ndarray) -> None:
    try:
        import IPython
        if IPython.get_ipython() is not None:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(grid_bgr.shape[1] / 100, grid_bgr.shape[0] / 100))
            plt.imshow(grid_bgr[:, :, ::-1])
            plt.axis("off")
            plt.tight_layout()
            plt.show()
            return
    except ImportError:
        pass
    cv2.imshow("batch", grid_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
