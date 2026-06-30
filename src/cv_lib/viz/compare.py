"""Visual comparison of ground truth annotations vs model predictions for a single image."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False


# BGR palette — cycles through classes
_PALETTE = [
    (0, 200, 0),    # green  — class 0
    (0, 0, 220),    # red
    (220, 180, 0),  # cyan-ish
    (200, 0, 200),  # magenta
    (0, 180, 220),  # yellow
    (100, 100, 220),# salmon
    (0, 220, 220),  # yellow-green
    (220, 100, 0),  # teal
]


def _color(class_id: int) -> tuple[int, int, int]:
    return _PALETTE[class_id % len(_PALETTE)]


def _draw_boxes(
    img: np.ndarray,
    boxes_xyxy: np.ndarray,
    class_ids: np.ndarray,
    class_names: list[str],
    scores: np.ndarray | None = None,
    line_thickness: int = 2,
) -> np.ndarray:
    """Draw bounding boxes on a copy of img. boxes_xyxy in absolute pixel coords."""
    out = img.copy()
    for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy.astype(int)):
        cid = int(class_ids[i])
        color = _color(cid)
        label = class_names[cid] if cid < len(class_names) else str(cid)
        if scores is not None:
            label = f"{label} {scores[i]:.2f}"

        cv2.rectangle(out, (x1, y1), (x2, y2), color, line_thickness)

        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        tag_y = max(y1 - 4, th + baseline)
        cv2.rectangle(out, (x1, tag_y - th - baseline), (x1 + tw, tag_y), color, -1)
        cv2.putText(out, label, (x1, tag_y - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def load_yolo_gt(
    label_path: str | Path,
    img_w: int,
    img_h: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse a YOLO-format .txt label file.

    Returns:
        boxes_xyxy: float32 array (N, 4) in absolute pixel coords
        class_ids:  int32 array (N,)
    """
    label_path = Path(label_path)
    if not label_path.exists():
        return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.int32)

    boxes, classes = [], []
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        cid, cx, cy, bw, bh = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        x1 = (cx - bw / 2) * img_w
        y1 = (cy - bh / 2) * img_h
        x2 = (cx + bw / 2) * img_w
        y2 = (cy + bh / 2) * img_h
        boxes.append([x1, y1, x2, y2])
        classes.append(cid)

    if not boxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.int32)

    return np.array(boxes, dtype=np.float32), np.array(classes, dtype=np.int32)


def _resolve_label_path(image_path: Path) -> Path:
    """
    Find the YOLO .txt label next to the image.
    Handles both flat layout (images/ sibling labels/) and same-dir layout.
    """
    same_dir = image_path.with_suffix(".txt")
    if same_dir.exists():
        return same_dir

    # images/ → labels/ sibling swap
    parts = image_path.parts
    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        label_parts = parts[:idx] + ("labels",) + parts[idx + 1:]
        sibling = Path(*label_parts).with_suffix(".txt")
        if sibling.exists():
            return sibling

    return same_dir  # will be flagged as missing by caller


def compare_gt_pred(
    image_path: str | Path,
    model_path: str | Path,
    class_names: list[str],
    conf_threshold: float = 0.25,
    label_path: str | Path | None = None,
    csv_path: str | Path | None = None,
    output_path: str | Path | None = None,
    show: bool = True,
) -> np.ndarray:
    """
    Side-by-side comparison: left = GT, right = model predictions.

    Args:
        image_path:     path to the image file (or, with ``csv_path``, just the
                        image name — the path is taken from the CSV's
                        ``image_path`` column when the given path is missing)
        model_path:     path to a .pt Ultralytics model
        class_names:    ordered list of class names matching the label indices
        conf_threshold: minimum confidence to show a prediction
        label_path:     explicit path to .txt label; auto-resolved if None
        csv_path:       CVAT CSV export to source GT boxes and the image path
                        from (instead of the YOLO dataset); the matching row's
                        path + any CVAT link column are printed
        output_path:    if set, saves the result image here
        show:           display with cv2.imshow (blocks until key press)

    Returns:
        side_by_side BGR image as np.ndarray
    """
    from ultralytics import YOLO  # imported lazily — not everyone installs it

    image_path = Path(image_path)

    # When a CVAT CSV is given, it is the source of truth for the image path and
    # the GT boxes (issue: compare straight from the export, not a YOLO dataset).
    record = None
    if csv_path is not None:
        from cv_lib.data.convert import cvat_csv_gt

        records = cvat_csv_gt(csv_path, class_names=class_names)
        record = records.get(image_path.name) or records.get(image_path.stem)
        if record and record.get("image_path") and not image_path.exists():
            image_path = Path(record["image_path"])

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    h, w = img.shape[:2]

    # --- Ground truth ---
    if record is not None:
        gt_boxes = (
            np.asarray(record["boxes"], dtype=np.float32)
            if record["boxes"]
            else np.zeros((0, 4), dtype=np.float32)
        )
        gt_classes = np.asarray(record["class_ids"], dtype=np.int32)
        meta = record.get("meta", {})
        link = next(
            (v for k, v in meta.items() if v and any(t in k.lower() for t in ("cvat", "url", "link"))),
            "",
        )
        print(f"[compare] {image_path}" + (f"  cvat: {link}" if link else ""))
    else:
        resolved_label = Path(label_path) if label_path else _resolve_label_path(image_path)
        gt_boxes, gt_classes = load_yolo_gt(resolved_label, w, h)
    gt_panel = _draw_boxes(img, gt_boxes, gt_classes, class_names)
    _add_panel_label(gt_panel, f"GT  ({len(gt_boxes)} boxes)")

    # --- Predictions ---
    model = YOLO(str(model_path))
    results = model(str(image_path), conf=conf_threshold, verbose=False)[0]

    pred_boxes = results.boxes.xyxy.cpu().numpy() if len(results.boxes) else np.zeros((0, 4))
    pred_classes = results.boxes.cls.cpu().numpy().astype(int) if len(results.boxes) else np.zeros(0, dtype=int)
    pred_scores = results.boxes.conf.cpu().numpy() if len(results.boxes) else np.zeros(0)

    pred_panel = _draw_boxes(img, pred_boxes, pred_classes, class_names, scores=pred_scores)
    _add_panel_label(pred_panel, f"Pred  ({len(pred_boxes)} boxes, conf≥{conf_threshold})")

    # --- Combine ---
    divider = np.full((h, 4, 3), 80, dtype=np.uint8)
    side_by_side = np.concatenate([gt_panel, divider, pred_panel], axis=1)

    if output_path:
        cv2.imwrite(str(output_path), side_by_side)

    if show:
        if _in_notebook():
            import matplotlib.pyplot as plt
            rgb = cv2.cvtColor(side_by_side, cv2.COLOR_BGR2RGB)
            fig, ax = plt.subplots(figsize=(min(side_by_side.shape[1] / 72, 24), 8))
            ax.imshow(rgb)
            ax.axis("off")
            fig.tight_layout()
            plt.show()
        else:
            win = f"compare — {image_path.name}"
            cv2.imshow(win, side_by_side)
            cv2.waitKey(0)
            cv2.destroyWindow(win)

    return side_by_side


def _add_panel_label(panel: np.ndarray, text: str) -> None:
    """Mutates panel in-place: adds a dark banner at the top."""
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 24), (30, 30, 30), -1)
    cv2.putText(panel, text, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
