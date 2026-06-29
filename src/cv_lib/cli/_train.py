"""`cvlib train` — train a YOLO model with seeds + config snapshot."""

from __future__ import annotations

import argparse

from loguru import logger

from cv_lib.cli._common import add_verbose

HELP = "Train a YOLO model (reproducible seeds + train_config.json snapshot)."

EPILOG = (
    "Examples:\n"
    "  cvlib train --model yolov8n.pt --data data.yaml --epochs 100\n"
    "  cvlib train --model yolov8s.pt --data data.yaml --imgsz 1280 --batch 8 --name run2\n"
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="Base .pt model or Ultralytics name (yolov8n.pt).")
    parser.add_argument("--data", required=True, help="Path to YOLO data.yaml.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs (default: 100).")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size (default: 640).")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument("--project", default="runs/train", help="Output project dir (default: runs/train).")
    parser.add_argument("--name", default="exp", help="Run name (default: exp).")
    parser.add_argument("--device", default=None, help="Device override, e.g. 'cpu', '0'.")
    add_verbose(parser)


def run(args: argparse.Namespace) -> None:
    from cv_lib.train import train

    logger.info("Training {} on {} for {} epochs …", args.model, args.data, args.epochs)
    train(
        model_path=args.model,
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        seed=args.seed,
        project=args.project,
        name=args.name,
        device=args.device,
    )
    logger.info("Done. Run dir: {}/{}", args.project, args.name)
