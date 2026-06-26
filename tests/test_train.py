"""Tests for cv_lib.train (seed determinism + config snapshot + train wrapper).

The actual model.train() is stubbed — we verify the reproducibility seeding and
that train() writes train_config.json before training and forwards the right
kwargs (model_path excluded, device only when set).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from cv_lib.train import set_seeds, train


def test_set_seeds_makes_random_and_numpy_deterministic():
    set_seeds(123)
    r1 = [random.random() for _ in range(3)]
    n1 = np.random.rand(3).tolist()

    set_seeds(123)
    r2 = [random.random() for _ in range(3)]
    n2 = np.random.rand(3).tolist()

    assert r1 == r2
    assert n1 == n2


def test_set_seeds_sets_torch_seed():
    import torch

    set_seeds(7)
    assert torch.initial_seed() == 7
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_train_writes_config_then_calls_model_train(tmp_path: Path, monkeypatch):
    captured: dict = {}

    class _FakeYOLO:
        def __init__(self, model_path):
            captured["init_path"] = model_path

        def train(self, **kwargs):
            captured["train_kwargs"] = kwargs
            return "results-sentinel"

    import ultralytics

    monkeypatch.setattr(ultralytics, "YOLO", _FakeYOLO)

    result = train(
        model_path="yolov8n.pt",
        data="data.yaml",
        epochs=3,
        imgsz=320,
        batch=8,
        seed=99,
        project=str(tmp_path),
        name="exp",
        extra_arg="x",
    )

    assert result == "results-sentinel"

    # config snapshot written before training, under <project>/<name>/
    cfg_path = tmp_path / "exp" / "train_config.json"
    assert cfg_path.exists()
    cfg = json.loads(cfg_path.read_text())
    assert cfg["model_path"] == "yolov8n.pt"
    assert cfg["epochs"] == 3
    assert cfg["seed"] == 99
    assert cfg["extra_arg"] == "x"

    # model_path must NOT be forwarded to model.train(); extra kwargs are
    assert "model_path" not in captured["train_kwargs"]
    assert captured["train_kwargs"]["data"] == "data.yaml"
    assert captured["train_kwargs"]["extra_arg"] == "x"
    # device omitted when not provided
    assert "device" not in captured["train_kwargs"]


def test_train_includes_device_when_set(tmp_path: Path, monkeypatch):
    captured: dict = {}

    class _FakeYOLO:
        def __init__(self, model_path):
            pass

        def train(self, **kwargs):
            captured.update(kwargs)
            return None

    import ultralytics

    monkeypatch.setattr(ultralytics, "YOLO", _FakeYOLO)

    train(
        model_path="yolov8n.pt",
        data="data.yaml",
        project=str(tmp_path),
        name="exp",
        device="cpu",
    )
    assert captured["device"] == "cpu"
