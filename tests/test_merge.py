"""Tests for cv_lib.data.merge (multi-dataset merge with class alignment)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml

from cv_lib.data.merge import DatasetSource, merge_datasets, source_from_root


def _make_dataset(root: Path, names: list[str], label_line: str) -> None:
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    cv2.imwrite(str(root / "images" / "f.jpg"), np.full((20, 20, 3), 10, np.uint8))
    (root / "labels" / "f.txt").write_text(label_line)
    (root / "data.yaml").write_text(yaml.safe_dump({"nc": len(names), "names": names}))


def test_merge_aligns_classes_by_name(tmp_path: Path):
    # dsA: [car, person]; dsB: [person, truck] → unified [car, person, truck]
    ds_a = tmp_path / "dsA"
    ds_b = tmp_path / "dsB"
    _make_dataset(ds_a, ["car", "person"], "1 0.5 0.5 0.2 0.2\n")  # person=1 in A
    _make_dataset(ds_b, ["person", "truck"], "0 0.5 0.5 0.2 0.2\n")  # person=0 in B
    out = tmp_path / "merged"

    report = merge_datasets(
        [source_from_root(ds_a), source_from_root(ds_b)], out, write_yaml=True
    )
    assert report.class_names == ["car", "person", "truck"]
    assert report.images == 2
    # both label files should now reference person = unified index 1
    a_id = (out / "labels" / "s0_f.txt").read_text().split()[0]
    b_id = (out / "labels" / "s1_f.txt").read_text().split()[0]
    assert a_id == "1" and b_id == "1"


def test_merge_prefixes_avoid_collision(tmp_path: Path):
    ds_a = tmp_path / "dsA"
    ds_b = tmp_path / "dsB"
    _make_dataset(ds_a, ["car"], "0 0.5 0.5 0.2 0.2\n")
    _make_dataset(ds_b, ["car"], "0 0.4 0.4 0.2 0.2\n")
    out = tmp_path / "merged"
    merge_datasets([source_from_root(ds_a), source_from_root(ds_b)], out)
    imgs = sorted(p.name for p in (out / "images").glob("*.jpg"))
    assert imgs == ["s0_f.jpg", "s1_f.jpg"]


def test_merge_writes_data_yaml(tmp_path: Path):
    ds = tmp_path / "ds"
    _make_dataset(ds, ["car", "person"], "0 0.5 0.5 0.2 0.2\n")
    out = tmp_path / "merged"
    report = merge_datasets([source_from_root(ds)], out)
    assert report.data_yaml is not None
    doc = yaml.safe_load(report.data_yaml.read_text())
    assert doc["nc"] == 2
    assert doc["names"] == ["car", "person"]


def test_source_dataclass_direct():
    src = DatasetSource(images_dir=Path("i"), labels_dir=Path("l"), class_names=["a"])
    assert src.class_names == ["a"]
