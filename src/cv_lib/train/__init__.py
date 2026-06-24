"""Thin training wrapper: seeds, config snapshot, model.train() call."""

from __future__ import annotations

import json
import random
from pathlib import Path


def set_seeds(seed: int = 42) -> None:
    """Set all relevant seeds for reproducibility."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _save_config(cfg: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_config.json").write_text(json.dumps(cfg, indent=2, default=str))


def train(
    model_path: str,
    data: str,
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    seed: int = 42,
    project: str = "runs/train",
    name: str = "exp",
    device: str | None = None,
    **kwargs,
) -> "ultralytics.engine.results.Results":
    """
    Train a YOLO model with reproducible seeds and config snapshot.

    Args:
        model_path:  path to .pt model or Ultralytics model name (e.g. 'yolov8n.pt')
        data:        path to YOLO data.yaml
        epochs:      number of training epochs
        imgsz:       input image size
        batch:       batch size
        seed:        random seed for reproducibility
        project:     output project directory
        name:        run name (subdirectory under project)
        device:      device override ('cpu', '0', etc.); auto if None
        **kwargs:    additional args forwarded to model.train()

    Returns:
        Ultralytics training Results object.
    """
    from ultralytics import YOLO

    set_seeds(seed)

    train_cfg: dict = {
        "model_path": model_path,
        "data": data,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "seed": seed,
        "project": project,
        "name": name,
        **kwargs,
    }
    if device is not None:
        train_cfg["device"] = device

    run_dir = Path(project) / name
    _save_config(train_cfg, run_dir)

    model = YOLO(model_path)

    train_kwargs = {k: v for k, v in train_cfg.items() if k != "model_path"}
    results = model.train(**train_kwargs)
    return results
