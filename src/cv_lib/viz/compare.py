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


def _model_class_names(results, model) -> list[str]:
    """Ordered class names from the model's own prediction (``names`` dict)."""
    src = getattr(results, "names", None) or getattr(model, "names", None)
    if isinstance(src, dict):
        return [src[i] for i in sorted(src)]
    return list(src) if src else []


def compare_gt_pred(
    image_path: str | Path,
    model_path,
    class_names: list[str] | None = None,
    conf_threshold: float = 0.25,
    label_path: str | Path | None = None,
    csv_path: str | Path | None = None,
    axis: str = "horizontal",
    output_path: str | Path | None = None,
    show: bool = True,
) -> np.ndarray:
    """
    GT vs model predictions in two panels (left/right or top/bottom).

    Args:
        image_path:     path to the image file (or, with ``csv_path``, just the
                        image name — the path is taken from the CSV's
                        ``image_path`` column when the given path is missing)
        model_path:     path to a ``.pt`` Ultralytics model **or** an already
                        loaded ``YOLO`` instance (pass the latter to reuse one
                        model across many calls instead of reloading each time)
        class_names:    ordered class names; if None, taken from the model's own
                        predictions (``results.names``)
        conf_threshold: minimum confidence to show a prediction
        label_path:     explicit path to .txt label; auto-resolved if None
        csv_path:       CVAT CSV export to source GT boxes and the image path
                        from (instead of the YOLO dataset); the matching row's
                        path + any CVAT link column are printed
        axis:           ``"horizontal"`` (side-by-side, default) or
                        ``"vertical"`` (stacked top/bottom — handier for wide
                        frames that would otherwise be tiny side-by-side)
        output_path:    if set, saves the result image here
        show:           display with cv2.imshow (blocks until key press)

    Returns:
        the combined BGR image as np.ndarray
    """
    # Accept a path (load once) or a pre-loaded model (reuse across calls).
    from cv_lib.data.autolabel import _load_model

    model = _load_model(model_path)
    image_path = Path(image_path)

    # When a CVAT CSV is given, it is the source of truth for the image path and
    # the GT boxes (compare straight from the export, not a YOLO dataset).
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

    # --- Predictions (single pass; also the fallback source of class names) ---
    results = model(str(image_path), conf=conf_threshold, verbose=False)[0]
    n_pred = len(results.boxes)
    pred_boxes = results.boxes.xyxy.cpu().numpy() if n_pred else np.zeros((0, 4))
    pred_classes = results.boxes.cls.cpu().numpy().astype(int) if n_pred else np.zeros(0, dtype=int)
    pred_scores = results.boxes.conf.cpu().numpy() if n_pred else np.zeros(0)

    names = list(class_names) if class_names else _model_class_names(results, model)

    # --- Ground truth ---
    if record is not None:
        gt_boxes = (
            np.asarray(record["boxes"], dtype=np.float32)
            if record["boxes"]
            else np.zeros((0, 4), dtype=np.float32)
        )
        gt_ids = []
        for label in record["labels"]:
            if label not in names:
                names.append(label)  # extend so the real CVAT label still renders
            gt_ids.append(names.index(label))
        gt_classes = np.asarray(gt_ids, dtype=np.int32)
        meta = record.get("meta", {})
        link = next(
            (v for k, v in meta.items() if v and any(t in k.lower() for t in ("cvat", "url", "link"))),
            "",
        )
        print(f"[compare] {image_path}" + (f"  cvat: {link}" if link else ""))
    else:
        resolved_label = Path(label_path) if label_path else _resolve_label_path(image_path)
        gt_boxes, gt_classes = load_yolo_gt(resolved_label, w, h)

    # Per-box confidence print (issue #22): one line per kept prediction.
    for (x1, y1, x2, y2), cid, score in zip(pred_boxes.astype(int), pred_classes, pred_scores):
        label = names[cid] if 0 <= cid < len(names) else str(cid)
        print(f"  [pred] {label:<14} conf={float(score):.3f}  box=({x1},{y1},{x2},{y2})")

    gt_panel = _draw_boxes(img, gt_boxes, gt_classes, names)
    _add_panel_label(gt_panel, f"GT  ({len(gt_boxes)} boxes)")
    pred_panel = _draw_boxes(img, pred_boxes, pred_classes, names, scores=pred_scores)
    _add_panel_label(pred_panel, f"Pred  ({n_pred} boxes, conf≥{conf_threshold})")

    # --- Combine: side-by-side or stacked ---
    if axis == "vertical":
        divider = np.full((4, w, 3), 80, dtype=np.uint8)
        combined = np.concatenate([gt_panel, divider, pred_panel], axis=0)
    else:
        divider = np.full((h, 4, 3), 80, dtype=np.uint8)
        combined = np.concatenate([gt_panel, divider, pred_panel], axis=1)

    if output_path:
        cv2.imwrite(str(output_path), combined)

    if show:
        if _in_notebook():
            import matplotlib.pyplot as plt
            rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
            fig, ax = plt.subplots(figsize=(min(combined.shape[1] / 72, 24), 8))
            ax.imshow(rgb)
            ax.axis("off")
            fig.tight_layout()
            plt.show()
        else:
            win = f"compare — {image_path.name}"
            cv2.imshow(win, combined)
            cv2.waitKey(0)
            cv2.destroyWindow(win)

    return combined


def _add_panel_label(panel: np.ndarray, text: str) -> None:
    """Mutates panel in-place: adds a dark banner at the top."""
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 24), (30, 30, 30), -1)
    cv2.putText(panel, text, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
