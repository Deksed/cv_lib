"""`cvlib qa` — flag suspicious-but-valid YOLO annotations."""

from __future__ import annotations

import argparse

from cv_lib.cli._common import add_verbose

HELP = "Audit YOLO labels for suspicious boxes (tiny/huge/aspect/duplicate/outlier)."


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("labels", help="Directory of YOLO .txt label files.")
    parser.add_argument(
        "--min-area", type=float, default=0.0005,
        help="Flag boxes with normalised area below this (default: 0.0005).",
    )
    parser.add_argument(
        "--max-area", type=float, default=0.9,
        help="Flag boxes with normalised area above this (default: 0.9).",
    )
    parser.add_argument(
        "--max-aspect", type=float, default=10.0,
        help="Flag boxes with side ratio above this (default: 10).",
    )
    parser.add_argument(
        "--count-z", type=float, default=3.0,
        help="Flag files with object count > mean + this*std (default: 3).",
    )
    add_verbose(parser)


def run(args: argparse.Namespace) -> int:
    from cv_lib.data.qa import audit_labels

    report = audit_labels(
        args.labels,
        min_box_area=args.min_area,
        max_box_area=args.max_area,
        max_aspect=args.max_aspect,
        count_z=args.count_z,
    )
    report.print()
    return 1 if report.findings else 0
