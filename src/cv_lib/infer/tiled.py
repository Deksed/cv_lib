"""Tiled / sliced inference for large images (SAHI-style).

Big frames downscaled to a detector's native input lose small objects. This
slices the image into overlapping tiles, runs the model on each, shifts the
boxes back into full-image coordinates, and de-duplicates overlaps with a
global NMS. The geometry (:func:`generate_tiles`, :func:`nms_numpy`) is pure
NumPy and unit-tested independently of any model.
"""

from __future__ import annotations

import numpy as np


def generate_tiles(
    width: int, height: int, tile: int = 640, overlap: float = 0.2
) -> list[tuple[int, int, int, int]]:
    """Tile a ``width × height`` frame into overlapping ``(x0, y0, x1, y1)`` boxes.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        tile: Tile side length in pixels.
        overlap: Fractional overlap between adjacent tiles (0–1).

    Returns:
        Pixel-coordinate tiles covering the whole frame. The last tile in each
        row/column is shifted inward so it never exceeds the image bounds.
    """
    if not 0 <= overlap < 1:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    step = max(1, int(tile * (1 - overlap)))

    def _starts(extent: int) -> list[int]:
        if extent <= tile:
            return [0]
        starts = list(range(0, extent - tile + 1, step))
        if starts[-1] != extent - tile:
            starts.append(extent - tile)
        return starts

    tiles: list[tuple[int, int, int, int]] = []
    for y0 in _starts(height):
        for x0 in _starts(width):
            tiles.append((x0, y0, min(x0 + tile, width), min(y0 + tile, height)))
    return tiles


def nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5) -> list[int]:
    """Greedy non-maximum suppression on ``xyxy`` boxes.

    Args:
        boxes: ``(N, 4)`` array of ``[x1, y1, x2, y2]``.
        scores: ``(N,)`` confidence scores.
        iou_threshold: Boxes overlapping a kept box above this IoU are dropped.

    Returns:
        Indices of kept boxes, highest score first.
    """
    boxes = np.asarray(boxes, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        union = areas[i] + areas[order[1:]] - inter
        iou = np.where(union > 0, inter / union, 0.0)
        order = order[1:][iou <= iou_threshold]
    return keep


def sliced_predict(
    model,
    image: np.ndarray,
    *,
    tile: int = 640,
    overlap: float = 0.2,
    conf: float = 0.25,
    nms_iou: float = 0.5,
    imgsz: int = 640,
    device: str | None = None,
) -> dict[str, np.ndarray]:
    """Run tiled inference and merge detections into full-image coordinates.

    Args:
        model: Ultralytics ``YOLO`` (or any object whose ``predict`` returns
            Ultralytics-style results with ``.boxes.xyxy/.cls/.conf``).
        image: ``H×W×3`` BGR image.
        tile: Tile side length.
        overlap: Fractional tile overlap.
        conf: Per-tile confidence threshold.
        nms_iou: IoU threshold for the global merge NMS.
        imgsz: Inference size passed to the model per tile.
        device: Optional device override.

    Returns:
        ``{"boxes": (N,4) xyxy, "scores": (N,), "classes": (N,)}`` in full-image
        pixel coordinates after global NMS.
    """
    h, w = image.shape[:2]
    tiles = generate_tiles(w, h, tile, overlap)

    all_boxes: list[np.ndarray] = []
    all_scores: list[float] = []
    all_classes: list[int] = []

    predict_kwargs: dict = {"conf": conf, "imgsz": imgsz, "verbose": False}
    if device is not None:
        predict_kwargs["device"] = device

    for x0, y0, x1, y1 in tiles:
        crop = image[y0:y1, x0:x1]
        results = model.predict(source=crop, **predict_kwargs)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            continue
        xyxy = np.asarray(boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else boxes.xyxy)
        cls = np.asarray(boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else boxes.cls)
        cf = np.asarray(boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else boxes.conf)
        xyxy = xyxy + np.array([x0, y0, x0, y0])  # shift to full-image coords
        for b, c, s in zip(xyxy, cls, cf):
            all_boxes.append(b)
            all_classes.append(int(c))
            all_scores.append(float(s))

    if not all_boxes:
        return {
            "boxes": np.zeros((0, 4), dtype=np.float32),
            "scores": np.zeros(0, dtype=np.float32),
            "classes": np.zeros(0, dtype=int),
        }

    boxes_arr = np.array(all_boxes, dtype=np.float32)
    scores_arr = np.array(all_scores, dtype=np.float32)
    classes_arr = np.array(all_classes, dtype=int)
    keep = nms_numpy(boxes_arr, scores_arr, nms_iou)
    return {
        "boxes": boxes_arr[keep],
        "scores": scores_arr[keep],
        "classes": classes_arr[keep],
    }
