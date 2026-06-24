"""Dataset utilities: YOLO format parsing, class distribution, split helpers."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import yaml


def load_dataset_yaml(path: str | Path) -> dict:
    """Load and return a YOLO data.yaml as a dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def class_names_from_yaml(path: str | Path) -> list[str]:
    """Extract ordered class names from a YOLO data.yaml."""
    data = load_dataset_yaml(path)
    names = data.get("names", {})
    if isinstance(names, dict):
        return [names[k] for k in sorted(names)]
    return list(names)


def iter_image_label_pairs(
    images_dir: str | Path,
    labels_dir: str | Path | None = None,
    extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp"),
) -> list[tuple[Path, Path]]:
    """
    Yield (image_path, label_path) pairs for a YOLO dataset split.

    If labels_dir is None, infers it by replacing 'images' with 'labels'
    in the images_dir path.

    Returns:
        Sorted list of (image_path, label_path) tuples.
        label_path may not exist if the image has no annotations.
    """
    images_dir = Path(images_dir)
    if labels_dir is None:
        parts = images_dir.parts
        if "images" in parts:
            idx = len(parts) - 1 - parts[::-1].index("images")
            labels_dir = Path(*parts[:idx], "labels", *parts[idx + 1:])
        else:
            labels_dir = images_dir

    labels_dir = Path(labels_dir)
    pairs = []
    for img in sorted(images_dir.iterdir()):
        if img.suffix.lower() in extensions:
            pairs.append((img, (labels_dir / img.stem).with_suffix(".txt")))
    return pairs


def class_distribution(
    labels_dir: str | Path,
    num_classes: int,
) -> np.ndarray:
    """
    Count annotations per class across all .txt files in labels_dir.

    Returns:
        int array of shape (num_classes,) with per-class box counts.
    """
    counts = np.zeros(num_classes, dtype=np.int64)
    for label_file in Path(labels_dir).glob("*.txt"):
        for line in label_file.read_text().splitlines():
            line = line.strip()
            if line:
                cid = int(line.split()[0])
                if 0 <= cid < num_classes:
                    counts[cid] += 1
    return counts


def data_root() -> Path:
    """Return DATA_ROOT env var as Path, raise if not set."""
    root = os.environ.get("DATA_ROOT")
    if not root:
        raise EnvironmentError("DATA_ROOT environment variable is not set.")
    return Path(root)
