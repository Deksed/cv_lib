"""FP/FN error visualization: find and render false positives, false negatives,
and worst-confidence predictions across a dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class ErrorEntry:
    image_path: Path
    error_type: str          # "FP", "FN", or "low_conf"
    box_xyxy: np.ndarray     # (4,) absolute pixel coords
    class_id: int
    conf: float              # 0.0 for GT-only (FN)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU between two boxes in xyxy format."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def find_errors(
    images_dir: str | Path,
    labels_dir: str | Path | None = None,
    model_path: str | Path | None = None,
    pred_labels_dir: str | Path | None = None,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.5,
    extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp"),
) -> list[ErrorEntry]:
    """
    Compare GT labels against model predictions (or pre-saved prediction labels)
    and return a list of FP, FN, and low-confidence detections.

    Provide either model_path (runs inference) or pred_labels_dir (reads saved YOLO txts).

    Args:
        images_dir:      directory of images
        labels_dir:      GT labels directory; inferred from images_dir if None
        model_path:      path to Ultralytics .pt model for live inference
        pred_labels_dir: directory of pre-saved YOLO prediction .txt files
        conf_threshold:  minimum confidence to count a prediction as positive
        iou_threshold:   IoU threshold to match a prediction to a GT box

    Returns:
        List of ErrorEntry sorted by error_type then confidence (ascending for low_conf).
    """
    if model_path is None and pred_labels_dir is None:
        raise ValueError("Provide either model_path or pred_labels_dir.")

    from cv_lib.viz.compare import load_yolo_gt

    images_dir = Path(images_dir)
    if labels_dir is None:
        parts = images_dir.parts
        if "images" in parts:
            idx = len(parts) - 1 - parts[::-1].index("images")
            labels_dir = Path(*parts[:idx], "labels", *parts[idx + 1:])
        else:
            labels_dir = images_dir
    labels_dir = Path(labels_dir)

    image_files = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in extensions)

    model = None
    if model_path is not None:
        from ultralytics import YOLO
        model = YOLO(str(model_path))

    errors: list[ErrorEntry] = []

    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        gt_label = (labels_dir / img_path.stem).with_suffix(".txt")
        gt_boxes, gt_cls = load_yolo_gt(gt_label, w, h)

        # --- get predictions ---
        if model is not None:
            result = model(str(img_path), conf=conf_threshold, verbose=False)[0]
            if len(result.boxes):
                pred_boxes = result.boxes.xyxy.cpu().numpy()
                pred_cls = result.boxes.cls.cpu().numpy().astype(int)
                pred_conf = result.boxes.conf.cpu().numpy()
            else:
                pred_boxes = np.zeros((0, 4), dtype=np.float32)
                pred_cls = np.zeros(0, dtype=int)
                pred_conf = np.zeros(0, dtype=np.float32)
        else:
            pred_label = (Path(pred_labels_dir) / img_path.stem).with_suffix(".txt")  # type: ignore[arg-type]
            if not pred_label.exists():
                pred_boxes = np.zeros((0, 4), dtype=np.float32)
                pred_cls = np.zeros(0, dtype=int)
                pred_conf = np.zeros(0, dtype=np.float32)
            else:
                pb, pc, pconf = [], [], []
                for line in pred_label.read_text().splitlines():
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cid = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:5])
                    conf = float(parts[5]) if len(parts) >= 6 else 1.0
                    if conf < conf_threshold:
                        continue
                    x1 = (cx - bw / 2) * w
                    y1 = (cy - bh / 2) * h
                    x2 = (cx + bw / 2) * w
                    y2 = (cy + bh / 2) * h
                    pb.append([x1, y1, x2, y2])
                    pc.append(cid)
                    pconf.append(conf)
                pred_boxes = np.array(pb, dtype=np.float32) if pb else np.zeros((0, 4), dtype=np.float32)
                pred_cls = np.array(pc, dtype=int)
                pred_conf = np.array(pconf, dtype=np.float32)

        # --- match preds to GT ---
        gt_matched = np.zeros(len(gt_boxes), dtype=bool)
        pred_matched = np.zeros(len(pred_boxes), dtype=bool)

        for pi in range(len(pred_boxes)):
            best_iou = 0.0
            best_gi = -1
            for gi in range(len(gt_boxes)):
                if gt_matched[gi]:
                    continue
                if pred_cls[pi] != gt_cls[gi]:
                    continue
                iou = _iou(pred_boxes[pi], gt_boxes[gi])
                if iou > best_iou:
                    best_iou = iou
                    best_gi = gi
            if best_iou >= iou_threshold and best_gi >= 0:
                gt_matched[best_gi] = True
                pred_matched[pi] = True

        # FP: unmatched predictions
        for pi in range(len(pred_boxes)):
            if not pred_matched[pi]:
                errors.append(ErrorEntry(
                    image_path=img_path,
                    error_type="FP",
                    box_xyxy=pred_boxes[pi],
                    class_id=int(pred_cls[pi]),
                    conf=float(pred_conf[pi]),
                ))

        # FN: unmatched GT
        for gi in range(len(gt_boxes)):
            if not gt_matched[gi]:
                errors.append(ErrorEntry(
                    image_path=img_path,
                    error_type="FN",
                    box_xyxy=gt_boxes[gi],
                    class_id=int(gt_cls[gi]),
                    conf=0.0,
                ))

    return sorted(errors, key=lambda e: (e.error_type, e.conf))


def render_errors(
    entries: list[ErrorEntry],
    class_names: list[str] | None = None,
    tile_size: tuple[int, int] = (320, 320),
    cols: int = 4,
    max_tiles: int = 32,
    output_path: str | Path | None = None,
    show: bool = True,
) -> np.ndarray:
    """
    Render a grid of error tiles: FP in red, FN in blue.

    Args:
        entries:      output of find_errors()
        class_names:  class name list
        tile_size:    (width, height) per tile
        cols:         grid columns
        max_tiles:    cap on number of tiles to render
        output_path:  save grid to this path if given
        show:         display with matplotlib or cv2

    Returns:
        Grid BGR ndarray.
    """
    class_names = class_names or []
    tw, th = tile_size
    entries = entries[:max_tiles]

    tiles: list[np.ndarray] = []
    for entry in entries:
        img = cv2.imread(str(entry.image_path))
        if img is None:
            tiles.append(np.zeros((th, tw, 3), dtype=np.uint8))
            continue

        tile = img.copy()
        x1, y1, x2, y2 = entry.box_xyxy.astype(int)
        color = (0, 0, 220) if entry.error_type == "FP" else (220, 0, 0)  # red FP, blue FN

        # crop around the box with 30px padding
        pad = 30
        cx_crop = max(0, min(x1, x2) - pad)
        cy_crop = max(0, min(y1, y2) - pad)
        cx2 = min(img.shape[1], max(x1, x2) + pad)
        cy2 = min(img.shape[0], max(y1, y2) + pad)
        crop = tile[cy_crop:cy2, cx_crop:cx2]
        if crop.size == 0:
            crop = tile

        # draw box on crop with offset coords
        ox, oy = x1 - cx_crop, y1 - cy_crop
        ox2, oy2 = x2 - cx_crop, y2 - cy_crop
        cv2.rectangle(crop, (ox, oy), (ox2, oy2), color, 2)

        cname = class_names[entry.class_id] if entry.class_id < len(class_names) else str(entry.class_id)
        if entry.error_type == "FP":
            label = f"FP {cname} {entry.conf:.2f}"
        else:
            label = f"FN {cname}"
        cv2.putText(crop, label, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        tiles.append(cv2.resize(crop, (tw, th)))

    if not tiles:
        return np.zeros((th, tw * cols, 3), dtype=np.uint8)

    rows = (len(tiles) + cols - 1) // cols
    while len(tiles) < rows * cols:
        tiles.append(np.zeros((th, tw, 3), dtype=np.uint8))

    grid_rows = [np.hstack(tiles[r * cols: (r + 1) * cols]) for r in range(rows)]
    grid = np.vstack(grid_rows)

    if output_path is not None:
        cv2.imwrite(str(output_path), grid)

    if show:
        try:
            import IPython
            if IPython.get_ipython() is not None:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(grid.shape[1] / 100, grid.shape[0] / 100))
                plt.imshow(grid[:, :, ::-1])
                plt.axis("off")
                plt.tight_layout()
                plt.show()
                return grid
        except ImportError:
            pass
        cv2.imshow("errors", grid)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return grid
