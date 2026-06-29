"""`cvlib threshold` — sweep confidence to pick a deployment operating point."""

from __future__ import annotations

import argparse

from loguru import logger

from cv_lib.cli._common import add_verbose

HELP = "Sweep confidence thresholds and report the one maximising F1 (or P/R)."


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="Path to Ultralytics .pt model.")
    parser.add_argument("--data", required=True, help="Path to YOLO data.yaml.")
    parser.add_argument(
        "--thresholds", nargs="+", type=float, default=None, metavar="C",
        help="Confidence values to test (default: 0.1 … 0.9).",
    )
    parser.add_argument(
        "--metric", choices=["f1", "precision", "recall"], default="f1",
        help="Metric to maximise (default: f1).",
    )
    parser.add_argument("--iou", type=float, default=0.6, help="NMS IoU for val (default: 0.6).")
    parser.add_argument(
        "--split", default="val", choices=["val", "test", "train"],
        help="Dataset split (default: val).",
    )
    parser.add_argument("--device", default=None, help="Device override, e.g. 'cpu', '0'.")
    add_verbose(parser)


def run(args: argparse.Namespace) -> None:
    from ultralytics import YOLO

    from cv_lib.metrics.threshold import sweep_threshold

    logger.info("Sweeping confidence thresholds for {} on {} …", args.model, args.data)
    model = YOLO(args.model)
    report = sweep_threshold(
        model,
        args.data,
        thresholds=args.thresholds,
        iou=args.iou,
        split=args.split,
        metric=args.metric,
        device=args.device,
    )
    report.print()
