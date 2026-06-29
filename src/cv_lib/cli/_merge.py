"""`cvlib merge` — merge several YOLO datasets under a unified taxonomy."""

from __future__ import annotations

import argparse

from cv_lib.cli._common import add_verbose

HELP = "Merge several YOLO datasets into one with a unified class taxonomy."

EPILOG = (
    "Each ROOT is a dataset dir with images/, labels/ and data.yaml (for class names).\n"
    "Classes are aligned by name; files are prefixed s<i>_ to avoid collisions.\n\n"
    "  cvlib merge dsA dsB dsC --out merged\n"
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("roots", nargs="+", help="Dataset roots (images/ + labels/ + data.yaml).")
    parser.add_argument("--out", required=True, metavar="DIR", help="Output dataset directory.")
    parser.add_argument(
        "--mode", choices=["copy", "move"], default="copy",
        help="How to place files (default: copy).",
    )
    parser.add_argument("--no-yaml", action="store_true", help="Do not write data.yaml.")
    add_verbose(parser)


def run(args: argparse.Namespace) -> None:
    from cv_lib.data.merge import merge_datasets, source_from_root

    sources = [source_from_root(root) for root in args.roots]
    report = merge_datasets(
        sources, args.out, mode=args.mode, write_yaml=not args.no_yaml
    )
    report.print()
