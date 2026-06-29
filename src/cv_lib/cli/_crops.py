"""`cvlib crops` — extract object crops from a YOLO dataset."""

from __future__ import annotations

import argparse

from cv_lib.cli._common import add_verbose, resolve_names

HELP = "Cut labelled objects out of a YOLO dataset into per-class crop folders."


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("images", help="Images directory.")
    parser.add_argument(
        "--labels", metavar="DIR",
        help="Labels directory (inferred from images path if omitted).",
    )
    parser.add_argument(
        "--out", default="crops", metavar="DIR",
        help="Output directory (default: crops).",
    )
    parser.add_argument(
        "--flat", action="store_true",
        help="Write all crops into one folder instead of per-class subfolders.",
    )
    parser.add_argument(
        "--pad", type=float, default=0.0,
        help="Fractional padding around each box, e.g. 0.1 (default: 0).",
    )

    names_group = parser.add_mutually_exclusive_group()
    names_group.add_argument(
        "--names", nargs="+", metavar="NAME", help="Class names for folder labels."
    )
    names_group.add_argument(
        "--data", metavar="YAML", help="data.yaml to read names from."
    )
    add_verbose(parser)


def run(args: argparse.Namespace) -> None:
    from cv_lib.data.crops import extract_crops

    class_names = resolve_names(args.names, args.data)
    report = extract_crops(
        args.images,
        labels_dir=args.labels,
        out_dir=args.out,
        per_class=not args.flat,
        pad=args.pad,
        class_names=class_names,
    )
    report.print()
