"""`cvlib autolabel` — bootstrap YOLO label drafts from a trained model."""

from __future__ import annotations

import argparse

from loguru import logger

from cv_lib.cli._common import add_verbose

HELP = "Pre-annotate a folder of images with a model into YOLO .txt drafts."


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("images", help="Directory of images (searched recursively).")
    parser.add_argument("--model", required=True, help="Path to Ultralytics .pt model.")
    parser.add_argument(
        "--out", required=True, metavar="DIR", help="Directory for YOLO .txt drafts."
    )
    parser.add_argument(
        "--conf", type=float, default=0.4, help="Confidence threshold (default: 0.4)."
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference size (default: 640).")
    parser.add_argument(
        "--save-conf", action="store_true", help="Append confidence as a 6th column."
    )
    parser.add_argument("--device", default=None, help="Device override, e.g. 'cpu', '0'.")
    add_verbose(parser)


def run(args: argparse.Namespace) -> None:
    from cv_lib.data.autolabel import autolabel

    n = autolabel(
        args.model,
        args.images,
        args.out,
        conf=args.conf,
        imgsz=args.imgsz,
        save_conf=args.save_conf,
        device=args.device,
    )
    logger.info("Auto-labelled {} image(s) → {}", n, args.out)
    print(f"Wrote {n} label draft(s) to {args.out}")
