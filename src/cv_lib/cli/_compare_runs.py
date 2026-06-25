"""`cvlib compare-runs` — compare training runs: configs + best metrics side-by-side."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from loguru import logger

from cv_lib.cli._common import add_verbose

HELP = "Compare training runs: config params + best metrics side-by-side."

EPILOG = (
    "Provide run dirs directly:  cvlib compare-runs runs/train/exp1 runs/train/exp2\n"
    "Or scan a project dir:      cvlib compare-runs --project runs/train"
)

# Config fields shown by default (in display order)
_DEFAULT_FIELDS = ["model_path", "epochs", "imgsz", "batch", "seed", "data"]

# Metric columns read from results.csv (Ultralytics names)
_METRIC_COLS = {
    "mAP50":    "metrics/mAP50(B)",
    "mAP50-95": "metrics/mAP50-95(B)",
    "precision": "metrics/precision(B)",
    "recall":   "metrics/recall(B)",
}


def _load_config(run_dir: Path) -> dict:
    cfg_path = run_dir / "train_config.json"
    if not cfg_path.exists():
        return {}
    return json.loads(cfg_path.read_text())


def _load_best_metrics(run_dir: Path) -> dict[str, float]:
    """Return the row with the highest mAP50 from results.csv, or {} if absent."""
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return {}

    map50_col = _METRIC_COLS["mAP50"]
    rows: list[dict] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        # Strip whitespace from header names (Ultralytics adds spaces)
        reader.fieldnames = [h.strip() for h in (reader.fieldnames or [])]
        for row in reader:
            rows.append({k.strip(): v.strip() for k, v in row.items()})

    if not rows:
        return {}

    def _map50(r: dict) -> float:
        try:
            return float(r.get(map50_col, 0) or 0)
        except ValueError:
            return 0.0

    best = max(rows, key=_map50)
    out: dict[str, float] = {}
    for label, col in _METRIC_COLS.items():
        if col in best:
            try:
                out[label] = float(best[col])
            except ValueError:
                pass
    return out


def _collect_runs(dirs: list[Path], project: Path | None) -> list[Path]:
    if project:
        found = sorted(
            p for p in project.iterdir()
            if p.is_dir() and (p / "train_config.json").exists()
        )
        if not found:
            logger.warning("No run dirs with train_config.json found under {}", project)
        return found
    return dirs


def _fmt_val(v: object) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v) if v is not None else "—"


def _print_table(runs: list[Path], fields: list[str]) -> None:
    configs = [_load_config(r) for r in runs]
    metrics = [_load_best_metrics(r) for r in runs]

    run_names = [r.name for r in runs]

    # Collect all rows: (section, key, [values])
    rows: list[tuple[str, str, list[str]]] = []

    for field in fields:
        vals = [_fmt_val(cfg.get(field)) for cfg in configs]
        rows.append(("config", field, vals))

    metric_keys = list(_METRIC_COLS.keys())
    has_any_metric = any(m for m in metrics)
    if has_any_metric:
        for key in metric_keys:
            vals = [_fmt_val(m.get(key)) for m in metrics]
            rows.append(("metrics", key, vals))

    # Column widths
    key_w = max(len(r[1]) for r in rows) + 2
    val_w = max(max(len(v) for v in r[2]) for r in rows)
    val_w = max(val_w, max(len(n) for n in run_names), 8)
    col_w = val_w + 2

    sep = "─" * (key_w + col_w * len(runs) + 2)

    # Header
    print()
    print(sep)
    header = f"{'':>{key_w}}"
    for name in run_names:
        header += f"  {name:>{val_w}}"
    print(header)
    print(sep)

    current_section = ""
    for section, key, vals in rows:
        if section != current_section:
            current_section = section
            print(f"  [{section}]")
        line = f"  {key:<{key_w - 2}}"
        for v in vals:
            line += f"  {v:>{val_w}}"
        print(line)

    print(sep)
    print()


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "runs", nargs="*", metavar="RUN_DIR",
        help="Run directories to compare (each must contain train_config.json).",
    )
    parser.add_argument(
        "--project", default=None, metavar="DIR",
        help="Parent directory — scans all subdirs with train_config.json.",
    )
    parser.add_argument(
        "--fields", nargs="+", default=_DEFAULT_FIELDS, metavar="FIELD",
        help=f"Config fields to show (default: {' '.join(_DEFAULT_FIELDS)}).",
    )
    add_verbose(parser)


def run(args: argparse.Namespace) -> None:
    project = Path(args.project) if args.project else None
    run_dirs = [Path(r) for r in args.runs]

    runs = _collect_runs(run_dirs, project)
    if not runs:
        logger.error("No run directories to compare. Pass RUN_DIR args or --project DIR.")
        sys.exit(1)

    missing = [r for r in runs if not r.exists()]
    if missing:
        for m in missing:
            logger.error("Directory not found: {}", m)
        sys.exit(1)

    logger.info("Comparing {} run(s):", len(runs))
    for r in runs:
        has_csv = (r / "results.csv").exists()
        logger.info("  {} {}", r, "(+ results.csv)" if has_csv else "(no results.csv)")

    _print_table(runs, args.fields)
