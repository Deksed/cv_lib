"""Tests for the `cvlib train` subcommand (train() is stubbed)."""

from __future__ import annotations

from cv_lib.cli import main


def test_train_cli_wires_arguments(monkeypatch):
    calls: dict = {}

    def _fake_train(**kwargs):
        calls.update(kwargs)
        return object()

    monkeypatch.setattr("cv_lib.train.train", _fake_train)

    code = main(
        [
            "train", "--model", "yolov8n.pt", "--data", "d.yaml",
            "--epochs", "5", "--imgsz", "320", "--batch", "8", "--name", "run2",
        ]
    )
    assert code == 0
    assert calls["model_path"] == "yolov8n.pt"
    assert calls["data"] == "d.yaml"
    assert calls["epochs"] == 5
    assert calls["imgsz"] == 320
    assert calls["batch"] == 8
    assert calls["name"] == "run2"
