#!/usr/bin/env python
"""
Run a YOLO model over a directory of images.

Outputs (choose one or both):
  --save-labels   write predictions as YOLO .txt files
  --save-vis      write side-by-side or annotated images

Usage examples:

  # Save YOLO labels only:
  python scripts/batch_infer.py --model best.pt --images dataset/images/val --save-labels

  # Save visualized images only:
  python scripts/batch_infer.py --model best.pt --images dataset/images/val --save-vis

  # Both, with class names from yaml:
  python scripts/batch_infer.py --model best.pt --images dataset/images/val \\
      --data dataset/data.yaml --save-labels --save-vis --out-dir runs/infer

  # Lower confidence, specific device:
  python scripts/batch_infer.py --model best.pt --images imgs/ \\
      --save-labels --conf 0.3 --device cpu
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cv_lib.data import class_names_from_yaml

_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")


def _collect_images(images_path: Path) -> list[Path]:
    if images_path.is_file():
        return [images_path]
    return sorted(p for p in images_path.rglob("*") if p.suffix.lower() in _EXTENSIONS)


def _boxes_to_yolo(boxes_xyxy: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """Convert xyxy pixel boxes → YOLO cx cy w h (normalised, 0–1)."""
    x1, y1, x2, y2 = boxes_xyxy[:, 0], boxes_xyxy[:, 1], boxes_xyxy[:, 2], boxes_xyxy[:, 3]
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return np.stack([cx, cy, w, h], axis=1)


def _draw_predictions(
    img_bgr: np.ndarray,
    boxes_xyxy: np.ndarray,
    class_ids: np.ndarray,
    confs: np.ndarray,
    class_names: list[str],
) -> np.ndarray:
    out = img_bgr.copy()
    for (x1, y1, x2, y2), cid, conf in zip(boxes_xyxy.astype(int), class_ids, confs):
        color = (0, 200, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{class_names[cid] if cid < len(class_names) else cid} {conf:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch inference with a YOLO model.")
    parser.add_argument("--model", required=True, help="Path to Ultralytics .pt model.")
    parser.add_argument(
        "--images",
        required=True,
        help="Path to image file or directory (searched recursively).",
    )

    names_group = parser.add_mutually_exclusive_group()
    names_group.add_argument(
        "--names", nargs="+", metavar="NAME",
        help="Class names in order, e.g. --names car person.",
    )
    names_group.add_argument(
        "--data", metavar="YAML",
        help="Path to YOLO data.yaml (reads names from it).",
    )

    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25).")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold (default: 0.45).")
    parser.add_argument("--device", default=None, help="Device override, e.g. 'cpu', '0'.")
    parser.add_argument(
        "--out-dir", default="runs/batch_infer",
        help="Root output directory (default: runs/batch_infer).",
    )
    parser.add_argument("--save-labels", action="store_true", help="Save predictions as YOLO .txt files.")
    parser.add_argument("--save-vis", action="store_true", help="Save annotated images.")
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Inference image size (default: 640).",
    )
    args = parser.parse_args()

    if not args.save_labels and not args.save_vis:
        parser.error("Specify at least one of --save-labels or --save-vis.")

    from ultralytics import YOLO

    model = YOLO(args.model)

    class_names: list[str] = []
    if args.data:
        class_names = class_names_from_yaml(args.data)
    elif args.names:
        class_names = args.names
    else:
        class_names = list(model.names.values()) if hasattr(model, "names") and model.names else []

    images_path = Path(args.images)
    image_files = _collect_images(images_path)
    if not image_files:
        print(f"No images found in {images_path}")
        return

    out_dir = Path(args.out_dir)
    labels_dir = out_dir / "labels"
    vis_dir = out_dir / "images"
    if args.save_labels:
        labels_dir.mkdir(parents=True, exist_ok=True)
    if args.save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    predict_kwargs: dict = {
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "verbose": False,
        "stream": True,
    }
    if args.device is not None:
        predict_kwargs["device"] = args.device

    print(f"Running inference on {len(image_files)} image(s) → {out_dir}")

    n_total = len(image_files)
    n_boxes = 0

    for i, (img_path, result) in enumerate(
        zip(image_files, model.predict(source=[str(p) for p in image_files], **predict_kwargs))
    ):
        boxes = result.boxes
        h, w = result.orig_shape

        if boxes is not None and len(boxes):
            boxes_xyxy = boxes.xyxy.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            n_boxes += len(boxes_xyxy)
        else:
            boxes_xyxy = np.zeros((0, 4), dtype=np.float32)
            class_ids = np.zeros(0, dtype=int)
            confs = np.zeros(0, dtype=np.float32)

        # --- YOLO labels ---
        if args.save_labels:
            label_out = labels_dir / f"{img_path.stem}.txt"
            if len(boxes_xyxy):
                yolo_coords = _boxes_to_yolo(boxes_xyxy, w, h)
                lines = []
                for cid, (cx, cy, bw, bh), conf in zip(class_ids, yolo_coords, confs):
                    lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {conf:.4f}")
                label_out.write_text("\n".join(lines))
            else:
                label_out.write_text("")

        # --- visualized image ---
        if args.save_vis:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                print(f"  Warning: could not read {img_path}, skipping vis.")
            else:
                vis = _draw_predictions(img_bgr, boxes_xyxy, class_ids, confs, class_names)
                vis_out = vis_dir / img_path.name
                cv2.imwrite(str(vis_out), vis)

        if (i + 1) % 50 == 0 or (i + 1) == n_total:
            print(f"  {i + 1}/{n_total} done")

    print(f"\nDone. {n_total} images, {n_boxes} total detections.")
    if args.save_labels:
        print(f"  Labels → {labels_dir}")
    if args.save_vis:
        print(f"  Images → {vis_dir}")


if __name__ == "__main__":
    main()
