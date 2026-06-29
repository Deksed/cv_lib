"""`cvlib remap` — merge / rename / drop classes across YOLO labels."""

from __future__ import annotations

import argparse

from cv_lib.cli._common import add_verbose

HELP = "Remap, merge or drop class ids across a directory of YOLO labels."

EPILOG = (
    "Examples:\n"
    "  cvlib remap labels/ --map 2=0 3=0 --out remapped/\n"
    "  cvlib remap labels/ --drop 5 --out remapped/\n"
    "  cvlib remap labels/ --map 1=0 --data data.yaml --names car --out remapped/\n"
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("labels", help="Directory of YOLO .txt label files.")
    parser.add_argument(
        "--map", nargs="+", default=[], metavar="OLD=NEW",
        help="Class id remap pairs, e.g. --map 2=0 3=0.",
    )
    parser.add_argument(
        "--drop", nargs="+", type=int, default=[], metavar="ID",
        help="Class ids to remove entirely (original ids).",
    )
    parser.add_argument(
        "--out", metavar="DIR",
        help="Output directory (overwrites in place if omitted).",
    )
    parser.add_argument(
        "--data", metavar="YAML", help="data.yaml to rewrite with --names."
    )
    parser.add_argument(
        "--names", nargs="+", metavar="NAME",
        help="New class names (written to data.yaml when --data is given).",
    )
    add_verbose(parser)


def run(args: argparse.Namespace) -> None:
    from cv_lib.data.remap import parse_mapping, remap_labels

    report = remap_labels(
        args.labels,
        mapping=parse_mapping(args.map),
        drop=set(args.drop),
        out_dir=args.out,
        class_names=args.names,
        data_yaml=args.data,
    )
    report.print()
