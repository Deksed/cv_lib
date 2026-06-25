"""`cvlib inspect` — dataset health check (corrupt images, missing labels, OOB boxes)."""

from __future__ import annotations

import argparse

from cv_lib.cli._common import add_verbose, apply_data_root, resolve_names

HELP = "Dataset health check: corrupt images, missing/empty labels, invalid boxes."


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("images", help="Path to images directory.")
    parser.add_argument(
        "--labels", default=None,
        help="Labels directory (inferred via images→labels swap if omitted).",
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

    parser.add_argument(
        "--num-classes", type=int, default=None,
        help="Expected number of classes (for out-of-range class id checks).",
    )
    add_verbose(parser)


def run(args: argparse.Namespace) -> None:
    from cv_lib.data.inspect import inspect_dataset

    class_names = resolve_names(args.names, args.data)
    num_classes = args.num_classes or (len(class_names) if class_names else None)

    images_dir = apply_data_root(args.images)
    labels_dir = apply_data_root(args.labels) if args.labels else None

    report = inspect_dataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        num_classes=num_classes,
        class_names=class_names,
    )
    report.print()
