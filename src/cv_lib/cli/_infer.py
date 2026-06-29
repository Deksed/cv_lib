"""`cvlib infer` — batch inference over a directory of images."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from cv_lib.cli._common import add_verbose

HELP = "Batch inference: write YOLO label .txt files and/or annotated images."

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


def add_arguments(parser: argparse.ArgumentParser) -> None:
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
    parser.add_argument(
        "--tiled", action="store_true",
        help="Sliced inference: split each image into overlapping tiles (small objects).",
    )
    parser.add_argument("--tile", type=int, default=640, help="Tile size for --tiled (default: 640).")
    parser.add_argument(
        "--overlap", type=float, default=0.2, help="Tile overlap fraction for --tiled (default: 0.2)."
    )
    add_verbose(parser)


def run(args: argparse.Namespace) -> None:
    if not args.save_labels and not args.save_vis:
        raise SystemExit("Specify at least one of --save-labels or --save-vis.")

    from ultralytics import YOLO

    from cv_lib.data import class_names_from_yaml

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
        logger.error("No images found in {}", images_path)
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
        "verbose": args.verbose,
        "stream": True,
    }
    if args.device is not None:
        predict_kwargs["device"] = args.device

    mode = "tiled" if args.tiled else "batch"
    logger.info("Running {} inference on {} image(s) → {}", mode, len(image_files), out_dir)

    n_total = len(image_files)
    n_boxes = 0

    def _save(img_path: Path, boxes_xyxy, class_ids, confs, w: int, h: int) -> None:
        if args.save_labels:
            label_out = labels_dir / f"{img_path.stem}.txt"
            if len(boxes_xyxy):
                yolo_coords = _boxes_to_yolo(boxes_xyxy, w, h)
                lines = [
                    f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {conf:.4f}"
                    for cid, (cx, cy, bw, bh), conf in zip(class_ids, yolo_coords, confs)
                ]
                label_out.write_text("\n".join(lines))
            else:
                label_out.write_text("")
        if args.save_vis:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                logger.warning("Could not read {}, skipping vis.", img_path)
            else:
                vis = _draw_predictions(img_bgr, boxes_xyxy, class_ids, confs, class_names)
                cv2.imwrite(str(vis_dir / img_path.name), vis)

    if args.tiled:
        from cv_lib.infer.tiled import sliced_predict

        for i, img_path in enumerate(image_files):
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                logger.warning("Could not read {}, skipping.", img_path)
                continue
            h, w = img_bgr.shape[:2]
            det = sliced_predict(
                model, img_bgr, tile=args.tile, overlap=args.overlap,
                conf=args.conf, nms_iou=args.iou, imgsz=args.imgsz, device=args.device,
            )
            n_boxes += len(det["boxes"])
            _save(img_path, det["boxes"], det["classes"], det["scores"], w, h)
            if (i + 1) % 50 == 0 or (i + 1) == n_total:
                logger.info("{}/{} done", i + 1, n_total)
    else:
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
            _save(img_path, boxes_xyxy, class_ids, confs, w, h)
            if (i + 1) % 50 == 0 or (i + 1) == n_total:
                logger.info("{}/{} done", i + 1, n_total)

    logger.info("Done. {} images, {} total detections.", n_total, n_boxes)
    if args.save_labels:
        logger.info("Labels → {}", labels_dir)
    if args.save_vis:
        logger.info("Images → {}", vis_dir)
