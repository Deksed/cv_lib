#!/usr/bin/env python
"""
Inference sanity check and benchmark for Ultralytics models.

Verifies a model loads and runs correctly, then benchmarks latency and
throughput across one or more models and input resolutions.

--- How to add models ---
  Pass .pt paths with --model. Three forms are accepted:

    Single model:
      python scripts/check_infer.py --model best.pt

    Compare several checkpoints:
      python scripts/check_infer.py --model runs/exp1/weights/best.pt runs/exp2/weights/best.pt

    All .pt files in a directory:
      python scripts/check_infer.py --model weights/

--- How to set resolutions ---
  Pass one or more sizes with --imgsz.
  Each size is tested for every model, so you get a full matrix:

    Single resolution (default):
      python scripts/check_infer.py --model best.pt --imgsz 640

    Sweep resolutions:
      python scripts/check_infer.py --model best.pt --imgsz 320 640 1280

Usage examples:

  # Quick smoke test (synthetic image, no dataset needed):
  python scripts/check_infer.py --model best.pt

  # Run on real images and save annotated outputs:
  python scripts/check_infer.py --model best.pt --source $DATA_ROOT/images/val --save

  # Compare two models at two resolutions, 200 timed runs each:
  python scripts/check_infer.py --model v1.pt v2.pt --imgsz 640 1280 --runs 200

  # CPU-only benchmark:
  python scripts/check_infer.py --model best.pt --device cpu

  # Headless (no DATA_ROOT needed):
  python scripts/check_infer.py --model best.pt --runs 100
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _setup_logging(verbose: bool) -> None:
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if verbose else "INFO", format="{level}: {message}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_models(paths: list[str]) -> list[Path]:
    """Expand paths: directory → all .pt inside; file → as-is."""
    result: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            found = sorted(p.glob("*.pt"))
            if not found:
                print(f"Warning: no .pt files found in {p}")
            result.extend(found)
        else:
            result.append(p)
    return result


def _collect_images(source: Optional[str]) -> Optional[list[Path]]:
    """Return image paths from a file or directory, or None to use synthetic input."""
    if source is None:
        return None
    p = Path(source)
    if p.is_file():
        return [p]
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    imgs = sorted(f for f in p.iterdir() if f.suffix.lower() in exts)
    return imgs if imgs else None


def _synthetic(imgsz: int) -> np.ndarray:
    return np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)


def _fmt(v: float, d: int = 1) -> str:
    return f"{v:.{d}f}"


def _peak_gpu_mb() -> Optional[float]:
    try:
        import torch
        if torch.cuda.is_available():
            mb = torch.cuda.max_memory_allocated() / 1024 ** 2
            torch.cuda.reset_peak_memory_stats()
            return mb
    except ImportError:
        pass
    return None


def _param_count(model) -> Optional[float]:
    try:
        return sum(p.numel() for p in model.model.parameters()) / 1e6
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Core benchmark
# ---------------------------------------------------------------------------

def benchmark_model(
    model_path: Path,
    imgsz: int,
    images: Optional[list[Path]],
    conf: float,
    warmup_runs: int,
    timed_runs: int,
    device: Optional[str],
    save: bool,
) -> dict:
    """
    Load model, run warmup, then timed inference.
    Returns a stats dict with latency, FPS, GPU memory, detection counts.
    """
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    params_m = _param_count(model)

    predict_kwargs: dict = {"imgsz": imgsz, "conf": conf, "verbose": False}
    if device is not None:
        predict_kwargs["device"] = device

    # Warmup with a synthetic image so the first real run isn't penalised
    dummy = _synthetic(imgsz)
    for _ in range(warmup_runs):
        model(dummy, **predict_kwargs)

    # Build source list for timed runs
    if images:
        src_list = [str(images[i % len(images)]) for i in range(timed_runs)]
        predict_kwargs["save"] = save
    else:
        src_list = [dummy] * timed_runs

    # Reset GPU memory counter right before timed block
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass

    latencies: list[float] = []
    total_boxes = 0
    class_counts: dict[int, int] = {}

    for src in src_list:
        t0 = time.perf_counter()
        results = model(src, **predict_kwargs)
        latencies.append((time.perf_counter() - t0) * 1000)

        r = results[0]
        if r.boxes is not None and len(r.boxes):
            total_boxes += len(r.boxes)
            for cid in r.boxes.cls.cpu().numpy().astype(int):
                class_counts[cid] = class_counts.get(cid, 0) + 1

    arr = np.array(latencies)
    return {
        "params_m": params_m,
        "mean_ms": float(arr.mean()),
        "p95_ms": float(np.percentile(arr, 95)),
        "fps": 1000.0 / float(arr.mean()),
        "gpu_mem_mb": _peak_gpu_mb(),
        "runs": timed_runs,
        "total_boxes": total_boxes,
        "class_counts": class_counts,
        "class_names": model.names,
    }


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

_COLS = ["Model", "imgsz", "Params(M)", "Mean(ms)", "P95(ms)", "FPS", "GPU(MB)", "Boxes/run"]


def _print_table(rows: list[dict]) -> None:
    if not rows:
        return
    col_model = max(len(r["model"]) for r in rows)
    widths = [max(col_model, len(_COLS[0])), 6, 9, 8, 7, 7, 7, 9]
    sep = "  "
    header = sep.join(h.ljust(w) for h, w in zip(_COLS, widths))
    rule = "─" * len(header)
    print()
    print(rule)
    print(header)
    print(rule)
    for r in rows:
        gpu = _fmt(r["gpu_mem_mb"]) if r["gpu_mem_mb"] is not None else "—"
        prm = _fmt(r["params_m"]) if r["params_m"] is not None else "—"
        cols = [
            r["model"].ljust(widths[0]),
            str(r["imgsz"]).ljust(widths[1]),
            prm.ljust(widths[2]),
            _fmt(r["mean_ms"]).ljust(widths[3]),
            _fmt(r["p95_ms"]).ljust(widths[4]),
            _fmt(r["fps"]).ljust(widths[5]),
            gpu.ljust(widths[6]),
            _fmt(r["total_boxes"] / r["runs"], 1).ljust(widths[7]),
        ]
        print(sep.join(cols))
    print(rule)


def _print_class_breakdown(label: str, imgsz: int, stats: dict) -> None:
    cc = stats["class_counts"]
    names = stats["class_names"]
    header = f"  [{label}  imgsz={imgsz}]"
    if not cc:
        print(f"{header}  no detections above conf threshold")
        return
    print(f"{header}  per-class detections across {stats['runs']} runs:")
    for cid in sorted(cc):
        name = names.get(cid, str(cid))
        print(f"    {name:<24}  total={cc[cid]:>6}  avg/run={cc[cid]/stats['runs']:>5.1f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inference sanity check and benchmark for Ultralytics models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Models  : --model best.pt  /  --model v1.pt v2.pt  /  --model weights/\n"
            "Sizes   : --imgsz 640  /  --imgsz 320 640 1280\n"
            "Source  : --source $DATA_ROOT/images/val  (omit to use synthetic input)"
        ),
    )
    parser.add_argument(
        "--model", nargs="+", required=True, metavar="PATH",
        help=".pt file(s) or a directory containing .pt files.",
    )
    parser.add_argument(
        "--source", default=None, metavar="PATH",
        help="Image file or directory. Omit to benchmark on a synthetic random image.",
    )
    parser.add_argument(
        "--imgsz", nargs="+", type=int, default=[640], metavar="N",
        help="Input resolution(s) to test (default: 640). Multiple values run a sweep.",
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Confidence threshold (default: 0.25).",
    )
    parser.add_argument(
        "--warmup", type=int, default=3,
        help="Warmup inference runs before timing (default: 3).",
    )
    parser.add_argument(
        "--runs", type=int, default=50,
        help="Timed inference runs per model × imgsz (default: 50).",
    )
    parser.add_argument(
        "--device", default=None,
        help="Device override: 'cpu', '0', 'cuda:0' (default: auto).",
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save annotated images (ignored for synthetic input).",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    _setup_logging(args.verbose)

    models = _collect_models(args.model)
    if not models:
        logger.error("No .pt model files found.")
        sys.exit(1)

    images = _collect_images(args.source)
    if args.source and images is None:
        logger.warning("No images found in {} — using synthetic input.", args.source)

    src_desc = (
        f"{len(images)} image(s) from {args.source}" if images else "synthetic random image"
    )
    logger.info("Models   : {}", ", ".join(m.name for m in models))
    logger.info("Sizes    : {}", args.imgsz)
    logger.info("Source   : {}", src_desc)
    logger.info("Runs     : {} warmup + {} timed  |  conf={}", args.warmup, args.runs, args.conf)

    table_rows: list[dict] = []
    breakdown: list[tuple[str, int, dict]] = []

    for model_path in models:
        for imgsz in args.imgsz:
            label = model_path.name
            logger.info("Checking {}  imgsz={} …", label, imgsz)
            try:
                stats = benchmark_model(
                    model_path=model_path,
                    imgsz=imgsz,
                    images=images,
                    conf=args.conf,
                    warmup_runs=args.warmup,
                    timed_runs=args.runs,
                    device=args.device,
                    save=args.save,
                )
                logger.info("  {} ms/frame  {} FPS", _fmt(stats["mean_ms"]), _fmt(stats["fps"]))
            except Exception as exc:
                logger.error("FAILED: {}", exc)
                continue

            table_rows.append({"model": label, "imgsz": imgsz, **stats})
            breakdown.append((label, imgsz, stats))

    _print_table(table_rows)
    print()
    for label, imgsz, stats in breakdown:
        _print_class_breakdown(label, imgsz, stats)
    print()


if __name__ == "__main__":
    main()
