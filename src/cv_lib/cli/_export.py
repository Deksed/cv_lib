"""`cvlib export` — export a YOLO model to ONNX (and optionally a TensorRT engine)."""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from cv_lib.cli._common import add_verbose

HELP = "Export a YOLO .pt model to ONNX (or build a TensorRT engine)."

EPILOG = """\
Examples:
  cvlib export best.pt --format onnx --out best.onnx
  cvlib export best.pt --format onnx --imgsz 1280 --no-dynamic
  cvlib export best.pt --format engine --fp16        # .pt -> .onnx -> .engine
  cvlib export model.onnx --format engine --out model.engine

TensorRT is not a project dependency — install it separately for --format engine.
"""


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "model",
        help="Path to a YOLO .pt model (or an .onnx file when --format engine).",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["onnx", "engine"],
        default="onnx",
        help="Export target: onnx, or engine (TensorRT). Default: onnx.",
    )
    parser.add_argument(
        "--out",
        "-o",
        default=None,
        metavar="PATH",
        help="Output file path (default: model path with the target suffix).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Square input size for export (default: 640).",
    )
    parser.add_argument(
        "--no-dynamic",
        action="store_true",
        help="Disable dynamic batch/spatial axes (ONNX).",
    )
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="Skip onnx-simplifier after ONNX export.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable FP16 precision when building the TensorRT engine.",
    )
    parser.add_argument(
        "--workspace",
        type=int,
        default=4,
        help="TensorRT builder workspace in GB (default: 4).",
    )
    add_verbose(parser)


def _export_to_onnx(args: argparse.Namespace, model_path: Path, out: Path) -> Path:
    from ultralytics import YOLO

    from cv_lib.export import export_onnx

    logger.info("Exporting {} -> ONNX ({}) …", model_path, out)
    model = YOLO(str(model_path))
    return export_onnx(
        model,
        out,
        input_shape=(1, 3, args.imgsz, args.imgsz),
        dynamic=not args.no_dynamic,
        simplify=not args.no_simplify,
    )


def run(args: argparse.Namespace) -> int:
    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")

    if args.format == "onnx":
        out = Path(args.out) if args.out else model_path.with_suffix(".onnx")
        result = _export_to_onnx(args, model_path, out)
        logger.success("ONNX written -> {}", result)
        print(result)
        return 0

    # --- format == engine (TensorRT) ---
    from cv_lib.export import export_trt

    suffix = model_path.suffix.lower()
    if suffix == ".pt":
        # No standalone ONNX given: build it first, then the engine.
        onnx_path = _export_to_onnx(args, model_path, model_path.with_suffix(".onnx"))
    elif suffix == ".onnx":
        onnx_path = model_path
    else:
        raise SystemExit(f"--format engine expects a .pt or .onnx input, got: {model_path.suffix}")

    out = Path(args.out) if args.out else onnx_path.with_suffix(".engine")
    logger.info("Building TensorRT engine {} -> {} …", onnx_path, out)
    result = export_trt(onnx_path, out, fp16=args.fp16, workspace_gb=args.workspace)
    logger.success("Engine written -> {}", result)
    print(result)
    return 0
