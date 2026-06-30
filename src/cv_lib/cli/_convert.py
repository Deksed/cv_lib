"""`cvlib convert` — CVAT XML / COCO JSON / CVAT CSV annotations → YOLO .txt files."""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from cv_lib.cli._common import add_verbose, resolve_names

HELP = "Convert CVAT XML / COCO JSON / CVAT CSV ↔ YOLO labels (COCO/VOC export too)."

EPILOG = (
    "Import (→ YOLO):\n"
    "  cvlib convert ann.json --out labels/\n"
    "  cvlib convert export.csv --out labels/\n"
    "Export (YOLO →):\n"
    "  cvlib convert labels/ --to coco     --images images/ --names car person --out ann.json\n"
    "  cvlib convert labels/ --to voc      --images images/ --names car person --out voc/\n"
    "  cvlib convert labels/ --to cvat-csv --images images/ --names car person --out gt.csv\n"
)


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
    parser.add_argument(
        "input", help="Annotation file (→ YOLO), or a YOLO labels dir (with --to)."
    )
    parser.add_argument(
        "--out", required=True, metavar="PATH",
        help="Output directory (YOLO/VOC) or .json file (COCO).",
    )
    parser.add_argument(
        "--format", choices=["auto", "cvat-xml", "coco", "cvat-csv", "voc"], default="auto",
        help="Input format for import (default: auto; 'voc' = a dir of Pascal VOC XML).",
    )
    parser.add_argument(
        "--to", choices=["yolo", "coco", "voc", "cvat-csv"], default="yolo",
        help="Target format. 'yolo' (default) imports; 'coco'/'voc'/'cvat-csv' export from YOLO.",
    )
    parser.add_argument(
        "--images", metavar="DIR",
        help="Images directory — required for --to coco/voc/cvat-csv (reads sizes).",
    )
    parser.add_argument(
        "--template", metavar="CSV",
        help="Template CVAT CSV: joins bookkeeping/link columns by image_name (--to cvat-csv).",
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
    class_names = resolve_names(args.names, args.data)

    # --- Export direction: YOLO labels → COCO / VOC / CVAT CSV ---
    if args.to in ("coco", "voc", "cvat-csv"):
        if not args.images:
            raise SystemExit(f"--images is required for --to {args.to} (needed to read image sizes).")
        if not class_names:
            raise SystemExit(f"--names or --data is required for --to {args.to}.")
        from cv_lib.data.convert import yolo_to_coco, yolo_to_cvat_csv, yolo_to_voc

        if args.to == "coco":
            coco = yolo_to_coco(args.images, args.input, args.out, class_names)
            logger.info(
                "YOLO → COCO: {} images, {} annotations → {}",
                len(coco["images"]), len(coco["annotations"]), args.out,
            )
        elif args.to == "voc":
            n = yolo_to_voc(args.images, args.input, args.out, class_names)
            logger.info("YOLO → VOC: wrote {} XML file(s) → {}", n, args.out)
        else:
            n = yolo_to_cvat_csv(
                args.images, args.input, args.out, class_names, template_csv=args.template
            )
            logger.info("YOLO → CVAT CSV: wrote {} row(s) → {}", n, args.out)
        return

    # --- Import direction: CVAT/COCO/VOC → YOLO ---
    from cv_lib.data.convert import (
        coco_json_to_yolo,
        cvat_csv_to_yolo,
        cvat_xml_to_yolo,
        voc_to_yolo,
    )

    input_path = Path(args.input)
    fmt = args.format if args.format != "auto" else _detect_format(input_path)

    converters = {
        "cvat-xml": cvat_xml_to_yolo,
        "coco": coco_json_to_yolo,
        "cvat-csv": cvat_csv_to_yolo,
        "voc": voc_to_yolo,  # input is a directory of VOC XML files
    }
    class_map = converters[fmt](input_path, args.out, class_names=class_names)

    logger.info("Converted {} ({}) → {}", input_path.name, fmt, args.out)
    print("Class map:")
    for name, idx in sorted(class_map.items(), key=lambda kv: kv[1]):
        print(f"  {idx}: {name}")
