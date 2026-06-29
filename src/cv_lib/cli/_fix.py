"""`cvlib fix` — auto-repair YOLO labels (clip OOB boxes, drop unsalvageable)."""

from __future__ import annotations

import argparse

from cv_lib.cli._common import add_verbose

HELP = "Repair YOLO labels: clip out-of-bounds boxes, drop unsalvageable ones."


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("labels", help="Directory of YOLO .txt label files.")
    parser.add_argument(
        "--out", metavar="DIR",
        help="Output directory (overwrites in place if omitted).",
    )
    parser.add_argument(
        "--num-classes", type=int, default=None,
        help="Drop boxes with class id outside [0, N).",
    )
    parser.add_argument(
        "--no-clip", action="store_true",
        help="Drop out-of-bounds boxes instead of clipping them.",
    )
    add_verbose(parser)


def run(args: argparse.Namespace) -> None:
    from cv_lib.data.repair import repair_labels

    report = repair_labels(
        args.labels,
        num_classes=args.num_classes,
        clip=not args.no_clip,
        out_dir=args.out,
    )
    report.print()
