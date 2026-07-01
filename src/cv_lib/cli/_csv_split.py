"""`cvlib csv-split` — train/val/test split computed on the flat CVAT CSV.

Writes per-split CSVs + a split_manifest.csv (image_name -> split); it does not
copy image/label files (use `cvlib split` for that).
"""

from __future__ import annotations

import argparse

from cv_lib.cli._common import add_verbose

HELP = "Split a CVAT CSV into train/val/test (random / temporal / per-camera)."

EPILOG = (
    "Methods:\n"
    "  random  — plain stratified split, one image at a time\n"
    "  temporal — frames within --gap of each other (by ts) share a split\n"
    "  camera  — temporal grouping done per camera\n"
    "\n"
    "Examples:\n"
    "  cvlib csv-split export.csv --out splits\n"
    "  cvlib csv-split export.csv --out splits --method temporal --gap 2\n"
    "  cvlib csv-split export.csv --out splits --method camera --ratios 0.8 0.2\n"
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("csv", help="Path to the flat CVAT CSV export.")
    parser.add_argument(
        "--out", metavar="DIR",
        help="Output directory for <split>.csv + split_manifest.csv "
             "(omit to only print the report).",
    )
    parser.add_argument(
        "--method", choices=["random", "temporal", "camera"], default="random",
        help="Grouping strategy (default: random).",
    )
    parser.add_argument(
        "--ratios", nargs="+", type=float, default=[0.8, 0.1, 0.1], metavar="R",
        help="Split ratios, 2 or 3 values summing to 1 (default: 0.8 0.1 0.1).",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42).")
    parser.add_argument(
        "--no-stratify", action="store_true",
        help="Disable stratification by dominant class.",
    )
    parser.add_argument(
        "--ts-column", default="ts", metavar="COL",
        help="Timestamp column for temporal/camera methods (default: ts).",
    )
    parser.add_argument(
        "--camera-column", default="camera", metavar="COL",
        help="Camera column for the camera method (default: camera).",
    )
    parser.add_argument(
        "--gap", type=float, default=1.0, metavar="GAP",
        help="Max ts spacing (in the ts column's own unit) to keep frames in "
             "one session (default: 1.0).",
    )
    parser.add_argument(
        "--label-column", metavar="COL",
        help="Label column; auto-detects instance_label/instance_lable if omitted.",
    )
    add_verbose(parser)


def run(args: argparse.Namespace) -> None:
    from cv_lib.data.csv_split import (
        camera_temporal_split_csv,
        random_split_csv,
        temporal_split_csv,
    )

    ratios = tuple(args.ratios)
    stratify = not args.no_stratify
    common = dict(
        ratios=ratios, seed=args.seed, stratify=stratify,
        label_column=args.label_column, out_dir=args.out,
    )

    if args.method == "random":
        report = random_split_csv(args.csv, **common)
    elif args.method == "temporal":
        report = temporal_split_csv(args.csv, ts_column=args.ts_column, gap=args.gap, **common)
    else:  # camera
        report = camera_temporal_split_csv(
            args.csv, camera_column=args.camera_column, ts_column=args.ts_column,
            gap=args.gap, **common,
        )
    report.print()
