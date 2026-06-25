"""`cvlib convert` — CVAT XML / COCO JSON / CVAT CSV annotations → YOLO .txt files."""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from cv_lib.cli._common import add_verbose, resolve_names

HELP = "Convert CVAT XML / COCO JSON / CVAT CSV annotations to YOLO .txt labels."


def _detect_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".xml":
        return "cvat-xml"
    if suffix == ".json":
        return "coco"
    if suffix == ".csv":
        return "cvat-csv"
    raise SystemExit(
        f"Cannot infer format from '{path.name}'. Pass --format cvat-xml|coco|cvat-csv."
    )


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", help="Path to CVAT .xml, COCO .json, or CVAT .csv.")
    parser.add_argument(
        "--out", required=True, metavar="DIR",
        help="Directory to write YOLO .txt files into.",
    )
    parser.add_argument(
        "--format", choices=["auto", "cvat-xml", "coco", "cvat-csv"], default="auto",
        help="Input format (default: auto — inferred from extension).",
    )

    names_group = parser.add_mutually_exclusive_group()
    names_group.add_argument(
        "--names", nargs="+", metavar="NAME",
        help="Class names in order; inferred from the file if omitted.",
    )
    names_group.add_argument(
        "--data", metavar="YAML",
        help="Path to YOLO data.yaml (reads names from it).",
    )
    add_verbose(parser)


def run(args: argparse.Namespace) -> None:
    from cv_lib.data.convert import (
        coco_json_to_yolo,
        cvat_csv_to_yolo,
        cvat_xml_to_yolo,
    )

    input_path = Path(args.input)
    fmt = args.format if args.format != "auto" else _detect_format(input_path)
    class_names = resolve_names(args.names, args.data)

    converters = {
        "cvat-xml": cvat_xml_to_yolo,
        "coco": coco_json_to_yolo,
        "cvat-csv": cvat_csv_to_yolo,
    }
    class_map = converters[fmt](input_path, args.out, class_names=class_names)

    logger.info("Converted {} ({}) → {}", input_path.name, fmt, args.out)
    print("Class map:")
    for name, idx in sorted(class_map.items(), key=lambda kv: kv[1]):
        print(f"  {idx}: {name}")
