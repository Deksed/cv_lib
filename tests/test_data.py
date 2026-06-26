"""Tests for cv_lib.data core helpers: YAML parsing, pair iteration, class
distribution counting, and DATA_ROOT resolution."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cv_lib.data import (
    class_distribution,
    class_names_from_yaml,
    data_root,
    iter_image_label_pairs,
    load_dataset_yaml,
)


def test_load_dataset_yaml_roundtrip(tmp_path: Path):
    y = tmp_path / "data.yaml"
    y.write_text("nc: 2\nnames: [car, person]\n")
    doc = load_dataset_yaml(y)
    assert doc["nc"] == 2
    assert doc["names"] == ["car", "person"]


def test_class_names_from_yaml_list_form(tmp_path: Path):
    y = tmp_path / "data.yaml"
    y.write_text("names: [car, person, bike]\n")
    assert class_names_from_yaml(y) == ["car", "person", "bike"]


def test_class_names_from_yaml_dict_form_is_ordered(tmp_path: Path):
    # dict mapping index -> name, intentionally out of order
    y = tmp_path / "data.yaml"
    y.write_text("names:\n  1: person\n  0: car\n  2: bike\n")
    assert class_names_from_yaml(y) == ["car", "person", "bike"]


def test_iter_image_label_pairs_infers_labels_dir(tmp_path: Path):
    images = tmp_path / "images" / "train"
    labels = tmp_path / "labels" / "train"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    (images / "a.jpg").write_bytes(b"x")
    (images / "b.png").write_bytes(b"x")
    (images / "notes.txt").write_text("ignored")  # non-image, skipped
    (labels / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n")

    pairs = iter_image_label_pairs(images)
    names = [(img.name, lbl) for img, lbl in pairs]

    assert [n for n, _ in names] == ["a.jpg", "b.png"]
    # labels dir inferred by swapping images -> labels
    a_label = dict((img.name, lbl) for img, lbl in pairs)["a.jpg"]
    assert a_label == labels / "a.txt"
    assert a_label.exists()
    # missing label path is still returned (may not exist)
    b_label = dict((img.name, lbl) for img, lbl in pairs)["b.png"]
    assert b_label == labels / "b.txt"
    assert not b_label.exists()


def test_iter_image_label_pairs_explicit_labels_dir(tmp_path: Path):
    images = tmp_path / "imgs"
    labels = tmp_path / "lbls"
    images.mkdir()
    labels.mkdir()
    (images / "f.jpg").write_bytes(b"x")
    pairs = iter_image_label_pairs(images, labels_dir=labels)
    assert pairs == [(images / "f.jpg", labels / "f.txt")]


def test_class_distribution_counts_and_ignores_out_of_range(tmp_path: Path):
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n1 0.5 0.5 0.1 0.1\n")
    (labels / "b.txt").write_text("0 0.5 0.5 0.1 0.1\n5 0.5 0.5 0.1 0.1\n")  # 5 OOB
    (labels / "empty.txt").write_text("\n")

    counts = class_distribution(labels, num_classes=2)
    assert isinstance(counts, np.ndarray)
    assert counts.tolist() == [2, 1]  # class 5 ignored


def test_data_root_reads_env(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    assert data_root() == Path(str(tmp_path))


def test_data_root_raises_when_unset(monkeypatch):
    monkeypatch.delenv("DATA_ROOT", raising=False)
    with pytest.raises(OSError, match="DATA_ROOT"):
        data_root()
