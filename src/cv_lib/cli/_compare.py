"""`cvlib compare` — GT vs prediction side-by-side for a single image."""

from __future__ import annotations

import argparse

from cv_lib.cli._common import add_verbose, apply_data_root, resolve_names

HELP = "Side-by-side ground truth vs prediction visualizer for one image."


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("image", help="Path to the image file.")
    parser.add_argument("--model", required=True, help="Path to Ultralytics .pt model.")

    names_group = parser.add_mutually_exclusive_group()
    names_group.add_argument(
        "--names", nargs="+", metavar="NAME",
        help="Class names in order; if omitted, taken from the model's predictions.",
    )
    names_group.add_argument(
        "--data", metavar="YAML",
        help="Path to YOLO data.yaml (reads names from it).",
    )

    parser.add_argument(
        "--label", default=None,
        help="Explicit path to YOLO .txt label (auto-resolved if omitted).",
    )
    parser.add_argument(
        "--csv", default=None,
        help="CVAT CSV export to source GT boxes and the image path from.",
    )
    parser.add_argument(
        "--axis", choices=["horizontal", "vertical"], default="horizontal",
        help="Panel layout: side-by-side (default) or stacked top/bottom.",
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Confidence threshold for predictions (default: 0.25).",
    )
    parser.add_argument(
        "--output", default=None,
        help="Save result to this path instead of (or in addition to) display.",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Do not open a display window (useful on headless servers).",
    )
    add_verbose(parser)


def run(args: argparse.Namespace) -> None:
    from cv_lib.viz.compare import compare_gt_pred

    image_path = apply_data_root(args.image)
    class_names = resolve_names(args.names, args.data)

    compare_gt_pred(
        image_path=image_path,
        model_path=args.model,
        class_names=class_names,
        conf_threshold=args.conf,
        label_path=args.label,
        csv_path=args.csv,
        axis=args.axis,
        output_path=args.output,
        show=not args.no_show,
    )
