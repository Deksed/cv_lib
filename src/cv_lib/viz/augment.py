"""Augmentation preview: original vs N augmented variants, boxes recomputed.

Applies an `albumentations` pipeline to one image several times and tiles the
results into a grid so you can eyeball a pipeline (and that boxes survive the
geometric transforms) *before* wiring it into training. Bounding boxes are
transformed alongside the pixels via ``BboxParams(format="yolo")``.

Returns an ``np.ndarray`` (BGR, uint8) like the other ``viz`` helpers. A custom
``transform`` must include ``bbox_params`` when boxes are supplied; the built-in
:func:`default_transform` already does.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from cv_lib.viz._utils import assemble_grid, display, draw_boxes_on, to_bgr_uint8

if TYPE_CHECKING:
    import albumentations as A
    import torch


def default_transform(seed: int | None = None) -> A.Compose:
    """A general-purpose preview pipeline (flip + photometric + affine).

    Geometric ops use a YOLO ``BboxParams`` so boxes are remapped and clipped;
    boxes left <20% visible after a transform are dropped. When ``seed`` is given
    the returned pipeline is seeded deterministically.
    """
    import albumentations as A

    transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.3),
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(0.0, 0.06),
                rotate=(-15, 15),
                p=0.7,
            ),
        ],
        bbox_params=A.BboxParams(
            format="yolo", label_fields=["class_labels"], min_visibility=0.2
        ),
    )
    if seed is not None:
        _seed_transform(transform, seed)
    return transform


def _seed_transform(transform: A.Compose, seed: int) -> None:
    """Seed an albumentations pipeline deterministically.

    albumentations >=1.4.21 (pinned in ``pyproject.toml``) carries its own RNG
    and exposes ``set_random_seed``, so we no longer touch the process-global
    ``random``/``numpy`` generators.
    """
    transform.set_random_seed(seed)


def _caption(tile: np.ndarray, text: str) -> None:
    """Draw a dark caption banner across the top of ``tile`` (in place)."""
    cv2.rectangle(tile, (0, 0), (tile.shape[1], 20), (30, 30, 30), -1)
    cv2.putText(
        tile, text, (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA
    )


def augment_preview(
    image: str | Path | np.ndarray | torch.Tensor,
    boxes_yolo: np.ndarray | None = None,
    *,
    transform: A.Compose | None = None,
    n: int = 8,
    class_names: list[str] | None = None,
    seed: int | None = 42,
    tile_size: tuple[int, int] = (320, 320),
    cols: int = 3,
    output_path: str | Path | None = None,
    show: bool = True,
) -> np.ndarray:
    """Render an original-vs-augmentations grid for a single image.

    Args:
        image: Path, BGR ``np.ndarray`` (H×W×C), or float ``torch.Tensor`` (C×H×W).
        boxes_yolo: ``(N, 5)`` array of ``[class_id, cx, cy, w, h]`` (YOLO
            normalised), or ``None`` for an unlabelled image.
        transform: An ``albumentations.Compose`` (must carry ``bbox_params`` when
            boxes are given). Defaults to :func:`default_transform`.
        n: Number of augmented variants to draw (the original is shown first).
        class_names: Names for box captions.
        seed: Base RNG seed; variant ``i`` uses ``seed + i`` for deterministic,
            varied output. ``None`` leaves the global RNG untouched.
        tile_size: ``(w, h)`` each tile is resized to.
        cols: Grid columns.
        output_path: If given, save the grid (BGR) here.
        show: Display in a notebook / window.

    Returns:
        The grid as an ``np.ndarray`` (BGR, uint8).
    """
    if cols < 1:
        raise ValueError(f"cols must be >= 1, got {cols}")

    if isinstance(image, (str, Path)):
        bgr = cv2.imread(str(image))
        if bgr is None:
            raise FileNotFoundError(f"Cannot read image: {image}")
    else:
        bgr = to_bgr_uint8(image)

    class_names = class_names or []
    transform = transform or default_transform()

    if boxes_yolo is not None:
        boxes_yolo = np.asarray(boxes_yolo, dtype=float).reshape(-1, 5)
        bboxes = boxes_yolo[:, 1:5].tolist()
        labels = boxes_yolo[:, 0].astype(int).tolist()
    else:
        bboxes, labels = [], []

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _tile(img_bgr: np.ndarray, bxs: list, lbls: list, caption: str) -> np.ndarray:
        drawn = img_bgr.copy()
        if bxs:
            drawn = draw_boxes_on(
                img_bgr, np.asarray(bxs, dtype=float), np.asarray(lbls, dtype=int), class_names
            )
        tile = cv2.resize(drawn, tile_size)
        _caption(tile, caption)
        return tile

    tiles = [_tile(bgr, bboxes, labels, "original")]
    for i in range(n):
        if seed is not None:
            _seed_transform(transform, seed + i)
        out = transform(image=rgb, bboxes=bboxes, class_labels=labels)
        aug_bgr = cv2.cvtColor(out["image"], cv2.COLOR_RGB2BGR)
        tiles.append(_tile(aug_bgr, list(out["bboxes"]), list(out["class_labels"]), f"aug {i + 1}"))

    grid = assemble_grid(tiles, cols, tile_size)

    if output_path is not None:
        cv2.imwrite(str(output_path), grid)
    if show:
        display(grid)

    return grid
