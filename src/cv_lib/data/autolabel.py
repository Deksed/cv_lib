"""Auto pre-annotation — bootstrap YOLO labels from a trained model.

Runs a detector over a folder of unlabelled images and writes YOLO ``.txt``
drafts that can be imported into CVAT for human correction. Pre-annotation
typically cuts manual labelling time several-fold.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")


def _boxes_to_yolo(boxes_xyxy: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """Convert xyxy pixel boxes → YOLO ``cx cy w h`` (normalised)."""
    x1, y1, x2, y2 = boxes_xyxy[:, 0], boxes_xyxy[:, 1], boxes_xyxy[:, 2], boxes_xyxy[:, 3]
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return np.stack([cx, cy, w, h], axis=1)


def _load_model(model):
    """Accept a path or an already-loaded model object."""
    if isinstance(model, (str, Path)):
        from ultralytics import YOLO

        return YOLO(str(model))
    return model


def autolabel(
    model,
    images_dir: str | Path,
    out_dir: str | Path,
    *,
    conf: float = 0.4,
    imgsz: int = 640,
    save_conf: bool = False,
    device: str | None = None,
    extensions: tuple[str, ...] = _IMAGE_EXTENSIONS,
) -> int:
    """Write YOLO label drafts for every image in ``images_dir``.

    Args:
        model: Ultralytics ``YOLO`` instance or path to a ``.pt`` file.
        images_dir: Directory of images (searched recursively).
        out_dir: Directory to write ``.txt`` drafts into (one per image,
            empty file when nothing is detected).
        conf: Confidence threshold for kept detections.
        imgsz: Inference image size.
        save_conf: Append the confidence as a 6th column (CVAT ignores it; handy
            for later filtering).
        device: Optional device override.
        extensions: Image extensions to include.

    Returns:
        Number of images processed.
    """
    model = _load_model(model)
    images_dir = Path(images_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in extensions)

    predict_kwargs: dict = {"conf": conf, "imgsz": imgsz, "verbose": False}
    if device is not None:
        predict_kwargs["device"] = device

    processed = 0
    for img_path in image_files:
        results = model.predict(source=str(img_path), **predict_kwargs)
        result = results[0]
        boxes = result.boxes
        h, w = result.orig_shape
        lines: list[str] = []
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
            cls = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.asarray(boxes.cls)
            cf = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf)
            yolo = _boxes_to_yolo(xyxy, w, h)
            for cid, (cx, cy, bw, bh), c in zip(cls.astype(int), yolo, cf):
                line = f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
                if save_conf:
                    line += f" {float(c):.4f}"
                lines.append(line)
        (out_dir / f"{img_path.stem}.txt").write_text("\n".join(lines))
        processed += 1

    return processed
