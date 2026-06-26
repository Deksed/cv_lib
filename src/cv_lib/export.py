"""ONNX and TensorRT export utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def export_onnx(
    model,
    path: str | Path,
    input_shape: tuple[int, int, int, int] = (1, 3, 640, 640),
    dynamic: bool = True,
    simplify: bool = True,
) -> Path:
    """
    Export an Ultralytics model to ONNX.

    Args:
        model:        Ultralytics YOLO instance
        path:         output .onnx file path
        input_shape:  (B, C, H, W) — the spatial size (H == W) is passed to the
                      exporter as imgsz; the batch/channel dims are informational
        dynamic:      export with dynamic batch / spatial axes
        simplify:     run onnx-simplifier after export

    Returns:
        Path to the exported .onnx file
    """
    path = Path(path)
    model.export(format="onnx", imgsz=input_shape[2], dynamic=dynamic, simplify=simplify)
    # Ultralytics saves next to the .pt — move if needed
    exported = Path(model.ckpt_path).with_suffix(".onnx")
    if exported != path:
        exported.rename(path)
    return path


def export_trt(
    onnx_path: str | Path,
    engine_path: str | Path,
    fp16: bool = True,
    workspace_gb: int = 4,
) -> Path:
    """
    Build a TensorRT engine from an ONNX file.

    Requires tensorrt to be installed (not in default dependencies).

    Args:
        onnx_path:    path to the .onnx file
        engine_path:  output .engine file path
        fp16:         enable FP16 precision
        workspace_gb: max GPU memory for the TRT builder

    Returns:
        Path to the built engine file
    """
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise ImportError("TensorRT is not installed. Install it separately.") from exc

    onnx_path = Path(onnx_path)
    engine_path = Path(engine_path)

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errors = [parser.get_error(i) for i in range(parser.num_errors)]
            raise RuntimeError(f"ONNX parse failed: {errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30))
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed.")

    engine_path.write_bytes(serialized)
    return engine_path


def validate_export(
    pytorch_output: np.ndarray,
    exported_output: np.ndarray,
    atol: float = 1e-4,
) -> None:
    """
    Assert that exported model output matches PyTorch reference within tolerance.

    Raises ValueError if max absolute difference exceeds atol.
    """
    diff = np.abs(pytorch_output - exported_output).max()
    if diff > atol:
        raise ValueError(
            f"Export validation failed: max abs diff {diff:.6f} > atol {atol}"
        )
