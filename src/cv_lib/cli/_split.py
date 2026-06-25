"""`cvlib split` — train/val/test split of a YOLO dataset + data.yaml."""

from __future__ import annotations

import argparse

from cv_lib.cli._common import add_verbose, resolve_names

HELP = "Split a YOLO dataset into train/val/test and write data.yaml."

EPILOG = (
    "Examples:\n"
    "  cvlib split data/images --labels data/labels --out data/dataset\n"
    "  cvlib split data/images --out ds --ratios 0.8 0.2 --names car person\n"
    "  cvlib split data/images --out ds --no-stratify --mode symlink\n"
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("images", help="Images directory.")
    parser.add_argument(
        "--labels", metavar="DIR",
        help="Labels directory (inferred from images path if omitted).",
    )
    parser.add_argument(
        "--out", required=True, metavar="DIR", help="Output dataset directory."
    )
    parser.add_argument(
        "--ratios", nargs="+", type=float, default=[0.8, 0.1, 0.1], metavar="R",
        help="Split ratios, 2 or 3 values summing to 1 (default: 0.8 0.1 0.1).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed (default: 42).")
    parser.add_argument(
        "--no-stratify", action="store_true",
        help="Disable stratification by dominant class.",
    )
    parser.add_argument(
        "--mode", choices=["copy", "symlink", "move"], default="copy",
        help="How to place files (default: copy).",
    )
    parser.add_argument("--no-yaml", action="store_true", help="Do not write data.yaml.")

    names_group = parser.add_mutually_exclusive_group()
    names_group.add_argument(
        "--names", nargs="+", metavar="NAME",
        help="Class names in order; inferred from labels if omitted.",
    )
    names_group.add_argument(
        "--data", metavar="YAML", help="Path to a YOLO data.yaml to read names from."
    )
    add_verbose(parser)


def run(args: argparse.Namespace) -> None:
    from cv_lib.data.split import train_val_test_split

    class_names = resolve_names(args.names, args.data)
    report = train_val_test_split(
        args.images,
        labels_dir=args.labels,
        out_dir=args.out,
        ratios=tuple(args.ratios),
        seed=args.seed,
        stratify_by_class=not args.no_stratify,
        class_names=class_names,
        mode=args.mode,
        write_yaml=not args.no_yaml,
    )
    report.print()
