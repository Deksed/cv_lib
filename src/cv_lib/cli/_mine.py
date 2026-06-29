"""`cvlib mine` — rank unlabelled images by labelling priority."""

from __future__ import annotations

import argparse

from cv_lib.cli._common import add_verbose

HELP = "Rank unlabelled images by model uncertainty (active-learning queue)."


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("images", help="Directory of images (searched recursively).")
    parser.add_argument("--model", required=True, help="Path to Ultralytics .pt model.")
    parser.add_argument(
        "--by", choices=["uncertainty", "low_conf", "num_detections"],
        default="uncertainty", help="Scoring strategy (default: uncertainty).",
    )
    parser.add_argument(
        "--conf", type=float, default=0.05,
        help="Low confidence floor so weak boxes still count (default: 0.05).",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference size (default: 640).")
    parser.add_argument(
        "--top", type=int, default=20, help="How many to print (default: 20)."
    )
    parser.add_argument("--device", default=None, help="Device override, e.g. 'cpu', '0'.")
    add_verbose(parser)


def run(args: argparse.Namespace) -> None:
    from cv_lib.data.mining import rank_for_labeling

    ranked = rank_for_labeling(
        args.model, args.images, by=args.by, conf=args.conf, imgsz=args.imgsz, device=args.device
    )
    print(f"\nTop {min(args.top, len(ranked))} of {len(ranked)} images (by {args.by}):")
    for path, score in ranked[: args.top]:
        print(f"  {score:6.3f}  {path}")
