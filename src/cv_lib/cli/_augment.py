"""`cvlib augment` — preview an augmentation pipeline on a single image."""

from __future__ import annotations

import argparse
from pathlib import Path

from cv_lib.cli._common import add_verbose, resolve_names

HELP = "Render an original-vs-augmentations grid for one image (boxes recomputed)."

EPILOG = (
    "Examples:\n"
    "  cvlib augment img.jpg                                  # unlabelled, default pipeline\n"
    "  cvlib augment img.jpg --labels img.txt --names car person\n"
    "  cvlib augment img.jpg --labels img.txt --data data.yaml -n 11 --out aug.png\n"
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("image", help="Path to the image to augment.")
    parser.add_argument(
        "--labels", metavar="TXT",
        help="YOLO label .txt for the image (auto-detected next to it if omitted).",
    )
    parser.add_argument(
        "-n", "--num", type=int, default=8, metavar="N",
        help="Number of augmented variants to draw (default: 8).",
    )
    parser.add_argument(
        "--out", metavar="PNG", default="augment_preview.png",
        help="Where to save the grid (default: augment_preview.png).",
    )
    parser.add_argument("--cols", type=int, default=3, help="Grid columns (default: 3).")
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base RNG seed; variant i uses seed+i (default: 42).",
    )

    names_group = parser.add_mutually_exclusive_group()
    names_group.add_argument(
        "--names", nargs="+", metavar="NAME", help="Class names in order (for captions)."
    )
    names_group.add_argument(
        "--data", metavar="YAML", help="YOLO data.yaml to read class names from."
    )
    add_verbose(parser)


def run(args: argparse.Namespace) -> int:
    import numpy as np

    from cv_lib.viz.augment import augment_preview

    image = Path(args.image)
    if not image.is_file():
        raise SystemExit(f"Image not found: {image}")

    # Labels: explicit --labels, else a sibling .txt with the same stem.
    label_path = Path(args.labels) if args.labels else image.with_suffix(".txt")
    boxes = None
    if label_path.is_file():
        rows = [
            [float(x) for x in line.split()]
            for line in label_path.read_text().splitlines()
            if line.strip()
        ]
        if rows:
            boxes = np.asarray(rows, dtype=float)
    elif args.labels:
        raise SystemExit(f"Label file not found: {label_path}")

    class_names = resolve_names(args.names, args.data)

    augment_preview(
        image,
        boxes,
        n=args.num,
        class_names=class_names,
        seed=args.seed,
        cols=args.cols,
        output_path=args.out,
        show=False,
    )
    print(f"Saved augmentation preview -> {args.out}")
    return 0
