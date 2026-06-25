"""Tests for cv_lib.data.split (train/val/test split + data.yaml)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from cv_lib.data.split import train_val_test_split


def _make_dataset(root: Path, n: int = 30, num_classes: int = 3) -> tuple[Path, Path]:
    images, labels = root / "images", root / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    for i in range(n):
        (images / f"img_{i:03d}.jpg").write_bytes(b"x")
        cid = i % num_classes  # balanced across classes
        (labels / f"img_{i:03d}.txt").write_text(f"{cid} 0.5 0.5 0.2 0.2\n")
    return images, labels


def test_counts_sum_and_dirs_created(tmp_path: Path):
    images, labels = _make_dataset(tmp_path, n=30)
    out = tmp_path / "ds"
    report = train_val_test_split(images, labels, out, ratios=(0.7, 0.2, 0.1))

    assert sum(report.counts.values()) == 30
    for split in ("train", "val", "test"):
        assert (out / "images" / split).is_dir()
        assert (out / "labels" / split).is_dir()
    # Total placed image files match the source count.
    placed = list((out / "images").rglob("*.jpg"))
    assert len(placed) == 30


def test_writes_data_yaml(tmp_path: Path):
    images, labels = _make_dataset(tmp_path, n=12)
    out = tmp_path / "ds"
    report = train_val_test_split(images, labels, out)

    assert report.data_yaml == out / "data.yaml"
    doc = yaml.safe_load(report.data_yaml.read_text(encoding="utf-8"))
    assert doc["nc"] == 3
    assert doc["names"] == ["0", "1", "2"]
    assert doc["train"] == "images/train"


def test_class_names_passthrough(tmp_path: Path):
    images, labels = _make_dataset(tmp_path, n=9)
    out = tmp_path / "ds"
    report = train_val_test_split(images, labels, out, class_names=["car", "person", "bike"])

    doc = yaml.safe_load(report.data_yaml.read_text(encoding="utf-8"))
    assert doc["names"] == ["car", "person", "bike"]
    assert report.num_classes == 3


def test_deterministic_for_same_seed(tmp_path: Path):
    images, labels = _make_dataset(tmp_path, n=30)
    a = train_val_test_split(images, labels, tmp_path / "a", seed=7)
    b = train_val_test_split(images, labels, tmp_path / "b", seed=7)

    def names(out: Path, split: str) -> set[str]:
        return {p.name for p in (out / "images" / split).glob("*.jpg")}

    for split in ("train", "val", "test"):
        assert names(a.out_dir, split) == names(b.out_dir, split)


def test_two_way_split_has_no_test(tmp_path: Path):
    images, labels = _make_dataset(tmp_path, n=20)
    out = tmp_path / "ds"
    report = train_val_test_split(images, labels, out, ratios=(0.8, 0.2))

    assert set(report.counts) == {"train", "val"}
    assert not (out / "images" / "test").exists()


def test_stratify_keeps_every_class_in_each_split(tmp_path: Path):
    images, labels = _make_dataset(tmp_path, n=30, num_classes=3)
    out = tmp_path / "ds"
    train_val_test_split(images, labels, out, ratios=(0.6, 0.2, 0.2), stratify_by_class=True)

    for split in ("train", "val", "test"):
        seen = {
            int(p.read_text().split()[0])
            for p in (out / "labels" / split).glob("*.txt")
        }
        assert seen == {0, 1, 2}, f"{split} missing classes: {seen}"


def test_ratios_must_sum_to_one(tmp_path: Path):
    images, labels = _make_dataset(tmp_path, n=10)
    with pytest.raises(ValueError):
        train_val_test_split(images, labels, tmp_path / "ds", ratios=(0.5, 0.2, 0.1))


def test_empty_dataset_raises(tmp_path: Path):
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    with pytest.raises(ValueError):
        train_val_test_split(tmp_path / "images", tmp_path / "labels", tmp_path / "ds")
