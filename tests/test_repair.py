"""Tests for cv_lib.data.repair (label auto-repair)."""

from __future__ import annotations

from pathlib import Path

from cv_lib.data.repair import repair_labels


def test_clip_out_of_bounds(tmp_path: Path):
    labels = tmp_path / "labels"
    labels.mkdir()
    # box runs off the right/bottom edge: cx=0.9 w=0.4 → x2=1.1
    (labels / "a.txt").write_text("0 0.9 0.9 0.4 0.4\n")
    out = tmp_path / "out"

    report = repair_labels(labels, out_dir=out, clip=True)
    assert report.boxes_clipped == 1
    assert report.boxes_kept == 1
    cid, cx, cy, w, h = (out / "a.txt").read_text().split()
    assert float(cx) <= 1.0 and float(cy) <= 1.0
    assert float(cx) + float(w) / 2 <= 1.0 + 1e-6


def test_no_clip_drops_oob(tmp_path: Path):
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "a.txt").write_text("0 0.9 0.9 0.4 0.4\n0 0.5 0.5 0.2 0.2\n")
    out = tmp_path / "out"
    report = repair_labels(labels, out_dir=out, clip=False)
    assert report.boxes_dropped == 1
    assert report.boxes_kept == 1


def test_drop_malformed_and_oob_class(tmp_path: Path):
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "a.txt").write_text(
        "0 0.5 0.5\n"  # too few fields
        "x 0.5 0.5 0.2 0.2\n"  # non-numeric
        "9 0.5 0.5 0.2 0.2\n"  # class id out of range
        "1 0.5 0.5 0.2 0.2\n"  # valid
    )
    out = tmp_path / "out"
    report = repair_labels(labels, out_dir=out, num_classes=2)
    assert report.boxes_kept == 1
    assert report.boxes_dropped == 3
    assert report.reasons.get("class id out of range") == 1


def test_preserves_extra_column(tmp_path: Path):
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "a.txt").write_text("0 0.5 0.5 0.2 0.2 0.97\n")
    out = tmp_path / "out"
    repair_labels(labels, out_dir=out)
    assert (out / "a.txt").read_text().split()[-1] == "0.97"
