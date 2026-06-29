"""Tests for cv_lib.data.remap (class remap / merge / drop)."""

from __future__ import annotations

from pathlib import Path

import pytest

from cv_lib.data.remap import parse_mapping, remap_labels


def _write_labels(d: Path) -> None:
    (d / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n2 0.7 0.7 0.1 0.1\n")
    (d / "b.txt").write_text("1 0.4 0.4 0.2 0.2\n")


def test_parse_mapping():
    assert parse_mapping(["2=0", "3=0", "5=1"]) == {2: 0, 3: 0, 5: 1}


def test_parse_mapping_invalid():
    with pytest.raises(ValueError, match="OLD=NEW"):
        parse_mapping(["2"])


def test_merge_and_drop(tmp_path: Path):
    labels = tmp_path / "labels"
    labels.mkdir()
    _write_labels(labels)
    out = tmp_path / "out"

    report = remap_labels(labels, mapping={1: 0, 2: 0}, drop={}, out_dir=out)

    # class 1 and 2 merged into 0 → a.txt has three class-0 boxes
    a_lines = (out / "a.txt").read_text().splitlines()
    assert [line.split()[0] for line in a_lines] == ["0", "0", "0"]
    assert report.boxes_remapped == 3  # 1+2 in a.txt, 1 in b.txt
    assert report.new_class_counts[0] == 4  # 3 in a + 1 in b


def test_drop_removes_boxes(tmp_path: Path):
    labels = tmp_path / "labels"
    labels.mkdir()
    _write_labels(labels)
    out = tmp_path / "out"

    report = remap_labels(labels, mapping={}, drop={2}, out_dir=out)
    assert report.boxes_dropped == 1
    first_ids = [line.split()[0] for line in (out / "a.txt").read_text().splitlines()]
    assert "2" not in first_ids


def test_geometry_preserved(tmp_path: Path):
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "a.txt").write_text("1 0.123456 0.234567 0.111 0.222 0.95\n")
    out = tmp_path / "out"

    remap_labels(labels, mapping={1: 0}, out_dir=out)
    assert (out / "a.txt").read_text().strip() == "0 0.123456 0.234567 0.111 0.222 0.95"


def test_rewrites_data_yaml(tmp_path: Path):
    import yaml

    labels = tmp_path / "labels"
    labels.mkdir()
    _write_labels(labels)
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("nc: 3\nnames: [a, b, c]\n")
    out = tmp_path / "out"

    report = remap_labels(
        labels, mapping={1: 0, 2: 0}, out_dir=out, class_names=["thing"], data_yaml=data_yaml
    )
    assert report.data_yaml is not None
    doc = yaml.safe_load(report.data_yaml.read_text())
    assert doc["nc"] == 1
    assert doc["names"] == ["thing"]
