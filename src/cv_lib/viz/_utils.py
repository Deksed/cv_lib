"""Shared internal helpers for the ``viz`` package.

These are the building blocks reused across :mod:`cv_lib.viz.batch`,
:mod:`cv_lib.viz.augment`, and friends: image normalisation, box drawing, grid
assembly, and display. They are internal (underscore-free names here, but the
module itself is ``_utils`` — not part of the public API) yet carry an explicit
intra-package contract so refactors stay localised.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    import torch


def to_bgr_uint8(img: np.ndarray | torch.Tensor) -> np.ndarray:
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


def draw_boxes_on(
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


def assemble_grid(
    tiles: list[np.ndarray], cols: int, tile_size: tuple[int, int]
) -> np.ndarray:
    """Pad ``tiles`` to a full ``cols``-wide grid and stack into one BGR image.

    ``tile_size`` is ``(width, height)`` — matching the ``viz`` convention.
    Missing cells are filled with black placeholders.
    """
    if cols < 1:
        raise ValueError(f"cols must be >= 1, got {cols}")
    tw, th = tile_size
    tiles = list(tiles)
    if not tiles:
        return np.zeros((th, tw * cols, 3), dtype=np.uint8)
    rows = (len(tiles) + cols - 1) // cols
    while len(tiles) < rows * cols:
        tiles.append(np.zeros((th, tw, 3), dtype=np.uint8))
    grid_rows = [np.hstack(tiles[r * cols : (r + 1) * cols]) for r in range(rows)]
    return np.vstack(grid_rows)


def display(grid_bgr: np.ndarray, window: str = "viz") -> None:
    """Show a BGR grid via matplotlib in Jupyter, else an OpenCV window."""
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
    cv2.imshow(window, grid_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
