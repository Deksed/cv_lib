"""`cvlib distribution` — class-frequency bar chart for a YOLO dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from cv_lib.cli._common import add_verbose, resolve_names

HELP = "Plot per-class box counts (optionally comparing train/val/test splits)."

EPILOG = (
    "Examples:\n"
    "  cvlib distribution data/dataset                 # auto-detect splits + data.yaml\n"
    "  cvlib distribution data/labels --names car person\n"
    "  cvlib distribution ds --out dist.png --sort --horizontal\n"
)

_SPLITS = ("train", "val", "test")


def _resolve_label_dirs(path: Path) -> dict[str, Path]:
    """Map a path to ``{group: labels_dir}``.

    Detects a dataset root with ``labels/<split>`` subdirs, or a directory whose
    immediate children are ``train``/``val``/``test``; otherwise treats ``path``
    itself as a single labels directory.
    """
    labels_root = path / "labels" if (path / "labels").is_dir() else path
    splits = {s: labels_root / s for s in _SPLITS if (labels_root / s).is_dir()}
    if splits:
        return splits
    return {path.name or "labels": path}


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "path",
        help="Dataset root (with labels/<split>) or a single labels directory.",
    )
    parser.add_argument(
        "--out", metavar="PNG", default="class_distribution.png",
        help="Where to save the chart (default: class_distribution.png).",
    )
    parser.add_argument("--title", default="Class distribution", help="Chart title.")
    parser.add_argument(
        "--sort", action="store_true", help="Order bars by descending total count."
    )
    parser.add_argument(
        "--horizontal", action="store_true", help="Draw horizontal bars."
    )
    parser.add_argument(
        "--log", action="store_true", help="Use a log scale on the count axis."
    )

    names_group = parser.add_mutually_exclusive_group()
    names_group.add_argument(
        "--names", nargs="+", metavar="NAME",
        help="Class names in order; inferred from labels (or data.yaml) if omitted.",
    )
    names_group.add_argument(
        "--data", metavar="YAML", help="YOLO data.yaml to read class names from."
    )
    add_verbose(parser)


def run(args: argparse.Namespace) -> int:
    from cv_lib.data import class_distribution
    from cv_lib.viz.distribution import _max_class_id, plot_class_distribution

    path = Path(args.path)
    if not path.exists():
        raise SystemExit(f"Path not found: {path}")

    groups = _resolve_label_dirs(path)

    # Names: explicit list / --data, else an auto-detected data.yaml in the root.
    class_names = resolve_names(args.names, args.data)
    if class_names is None and (path / "data.yaml").is_file():
        from cv_lib.data import class_names_from_yaml

        class_names = class_names_from_yaml(path / "data.yaml")

    if class_names is not None:
        num_classes = len(class_names)
    else:
        num_classes = max((_max_class_id(d) for d in groups.values()), default=-1) + 1
    if num_classes <= 0:
        raise SystemExit(f"No labels found under {path}.")

    names = list(class_names) if class_names is not None else [str(i) for i in range(num_classes)]

    # Print a counts table (groups as columns) before saving the figure.
    counts = {g: class_distribution(d, num_classes) for g, d in groups.items()}
    header = f"{'class':<20}" + "".join(f"{g:>10}" for g in counts)
    print(header)
    print("-" * len(header))
    for i in range(num_classes):
        row = f"{names[i]:<20}" + "".join(f"{int(counts[g][i]):>10}" for g in counts)
        print(row)
    totals = f"{'TOTAL':<20}" + "".join(f"{int(counts[g].sum()):>10}" for g in counts)
    print("-" * len(header))
    print(totals)

    plot_class_distribution(
        {g: str(d) for g, d in groups.items()},
        class_names=names,
        num_classes=num_classes,
        title=args.title,
        sort=args.sort,
        horizontal=args.horizontal,
        log_scale=args.log,
        output_path=args.out,
    )
    print(f"\nSaved chart -> {args.out}")
    return 0
