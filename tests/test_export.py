"""Tests for cv_lib.export and the `cvlib export` subcommand.

The heavy paths (Ultralytics model load, real ONNX/TRT export) are stubbed: we
assert the CLI wires the right arguments into export_onnx / export_trt and picks
sane default output paths, plus exercise the pure validate_export numerics.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cv_lib.cli import main
from cv_lib.export import validate_export


# --------------------------------------------------------------------------- #
# validate_export (pure numeric helper)
# --------------------------------------------------------------------------- #
def test_validate_export_within_tolerance():
    a = np.zeros((2, 3), dtype=np.float32)
    b = a + 1e-6
    validate_export(a, b, atol=1e-4)  # must not raise


def test_validate_export_exceeds_tolerance():
    a = np.zeros((2, 3), dtype=np.float32)
    b = a.copy()
    b[0, 0] = 0.5
    with pytest.raises(ValueError, match="max abs diff"):
        validate_export(a, b, atol=1e-4)


# --------------------------------------------------------------------------- #
# CLI: onnx export
# --------------------------------------------------------------------------- #
def test_export_onnx_invokes_export_onnx(tmp_path: Path, monkeypatch):
    model_file = tmp_path / "best.pt"
    model_file.write_bytes(b"stub")

    calls: dict = {}

    class _FakeYOLO:
        def __init__(self, path):
            calls["yolo_path"] = path

    def _fake_export_onnx(model, path, *, input_shape, dynamic, simplify):
        calls["onnx"] = {
            "path": Path(path),
            "input_shape": input_shape,
            "dynamic": dynamic,
            "simplify": simplify,
        }
        Path(path).write_bytes(b"onnx")
        return Path(path)

    import ultralytics

    monkeypatch.setattr(ultralytics, "YOLO", _FakeYOLO)
    monkeypatch.setattr("cv_lib.export.export_onnx", _fake_export_onnx)

    code = main(["export", str(model_file), "--imgsz", "1280", "--no-dynamic"])
    assert code == 0

    # default output path = model with .onnx suffix
    assert calls["onnx"]["path"] == model_file.with_suffix(".onnx")
    assert calls["onnx"]["input_shape"] == (1, 3, 1280, 1280)
    assert calls["onnx"]["dynamic"] is False
    assert calls["onnx"]["simplify"] is True


def test_export_missing_model_exits(tmp_path: Path):
    with pytest.raises(SystemExit):
        main(["export", str(tmp_path / "nope.pt")])


# --------------------------------------------------------------------------- #
# CLI: engine export (.pt -> .onnx -> .engine)
# --------------------------------------------------------------------------- #
def test_export_engine_from_pt_builds_onnx_then_trt(tmp_path: Path, monkeypatch):
    model_file = tmp_path / "best.pt"
    model_file.write_bytes(b"stub")

    calls: dict = {}

    class _FakeYOLO:
        def __init__(self, path):
            pass

    def _fake_export_onnx(model, path, **kwargs):
        Path(path).write_bytes(b"onnx")
        calls["onnx_path"] = Path(path)
        return Path(path)

    def _fake_export_trt(onnx_path, engine_path, *, fp16, workspace_gb):
        calls["trt"] = {
            "onnx_path": Path(onnx_path),
            "engine_path": Path(engine_path),
            "fp16": fp16,
            "workspace_gb": workspace_gb,
        }
        return Path(engine_path)

    import ultralytics

    monkeypatch.setattr(ultralytics, "YOLO", _FakeYOLO)
    monkeypatch.setattr("cv_lib.export.export_onnx", _fake_export_onnx)
    monkeypatch.setattr("cv_lib.export.export_trt", _fake_export_trt)

    code = main(["export", str(model_file), "--format", "engine", "--fp16"])
    assert code == 0

    # .pt first becomes .onnx, which then feeds the engine build
    assert calls["onnx_path"] == model_file.with_suffix(".onnx")
    assert calls["trt"]["onnx_path"] == model_file.with_suffix(".onnx")
    assert calls["trt"]["engine_path"] == model_file.with_suffix(".engine")
    assert calls["trt"]["fp16"] is True
    assert calls["trt"]["workspace_gb"] == 4


def test_export_engine_rejects_bad_suffix(tmp_path: Path):
    bad = tmp_path / "model.bin"
    bad.write_bytes(b"x")
    with pytest.raises(SystemExit):
        main(["export", str(bad), "--format", "engine"])
