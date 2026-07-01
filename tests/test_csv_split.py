"""Tests for cv_lib.data.csv_split (random / temporal / camera CSV splits)."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from cv_lib.data.csv_split import (
    camera_temporal_split_csv,
    random_split_csv,
    temporal_split_csv,
)

_COLUMNS = ("image_name", "image_width", "image_height", "instance_label", "ts", "camera")


def _write_csv(path: Path, rows: list[dict]) -> Path:
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _make_csv(
    path: Path, n: int = 30, num_classes: int = 3, *, cameras: int = 1, ts_step: float = 100.0
) -> Path:
    """One instance row per image; classes cycle; ts increases per image."""
    rows = []
    for i in range(n):
        rows.append({
            "image_name": f"img_{i:03d}.jpg",
            "image_width": "640",
            "image_height": "480",
            "instance_label": f"cls{i % num_classes}",
            "ts": f"{i * ts_step:.1f}",
            "camera": f"cam{i % cameras}",
        })
    return _write_csv(path, rows)


def test_random_assigns_every_image_once(tmp_path: Path):
    csv_path = _make_csv(tmp_path / "e.csv", n=30)
    report = random_split_csv(csv_path, ratios=(0.7, 0.2, 0.1))

    assert len(report.assignment) == 30
    assert sum(report.counts.values()) == 30
    assert set(report.counts) == {"train", "val", "test"}


def test_random_deterministic_for_same_seed(tmp_path: Path):
    csv_path = _make_csv(tmp_path / "e.csv", n=40)
    a = random_split_csv(csv_path, seed=7)
    b = random_split_csv(csv_path, seed=7)
    assert a.assignment == b.assignment


def test_random_stratify_keeps_classes_in_each_split(tmp_path: Path):
    csv_path = _make_csv(tmp_path / "e.csv", n=30, num_classes=3)
    report = random_split_csv(csv_path, ratios=(0.6, 0.2, 0.2), stratify=True)

    strata = {"train": set(), "val": set(), "test": set()}
    for i in range(30):
        split = report.assignment[f"img_{i:03d}.jpg"]
        strata[split].add(i % 3)
    for split, seen in strata.items():
        assert seen == {0, 1, 2}, f"{split} missing classes: {seen}"


def test_two_way_split_has_no_test(tmp_path: Path):
    csv_path = _make_csv(tmp_path / "e.csv", n=20)
    report = random_split_csv(csv_path, ratios=(0.8, 0.2))
    assert set(report.counts) == {"train", "val"}
    assert "test" not in report.counts


def test_temporal_keeps_a_session_in_one_split(tmp_path: Path):
    # 4 tight bursts of 5 frames (ts 0..4), each burst 1000s apart -> 4 sessions.
    rows = []
    idx = 0
    for burst in range(4):
        for k in range(5):
            rows.append({
                "image_name": f"img_{idx:03d}.jpg",
                "image_width": "640", "image_height": "480",
                "instance_label": "cls0",
                "ts": f"{burst * 1000 + k:.1f}",
                "camera": "cam0",
            })
            idx += 1
    csv_path = _write_csv(tmp_path / "e.csv", rows)

    report = temporal_split_csv(csv_path, gap=1.0, ratios=(0.5, 0.5), stratify=False)

    # Every 5-frame burst must land entirely in a single split.
    for burst in range(4):
        splits = {report.assignment[f"img_{burst * 5 + k:03d}.jpg"] for k in range(5)}
        assert len(splits) == 1, f"burst {burst} was split across {splits}"


def test_camera_isolates_same_ts_across_cameras(tmp_path: Path):
    # Two cameras, identical timestamps; each camera is one tight session.
    rows = []
    for cam in ("camA", "camB"):
        for k in range(6):
            rows.append({
                "image_name": f"{cam}_{k}.jpg",
                "image_width": "640", "image_height": "480",
                "instance_label": "cls0",
                "ts": f"{k:.1f}",
                "camera": cam,
            })
    csv_path = _write_csv(tmp_path / "e.csv", rows)

    report = camera_temporal_split_csv(csv_path, gap=1.0, ratios=(0.5, 0.5), stratify=False)

    for cam in ("camA", "camB"):
        splits = {report.assignment[f"{cam}_{k}.jpg"] for k in range(6)}
        assert len(splits) == 1, f"{cam} session was split across {splits}"


def test_writes_split_csvs_and_manifest(tmp_path: Path):
    csv_path = _make_csv(tmp_path / "e.csv", n=20)
    out = tmp_path / "splits"
    report = random_split_csv(csv_path, out_dir=out)

    assert report.out_dir == out
    for split in ("train", "val", "test"):
        assert report.files[split].exists()
    manifest = report.files["manifest"]
    assert manifest.exists()

    # Manifest rows match the assignment; per-split CSVs partition all rows.
    with open(manifest, newline="", encoding="utf-8-sig") as f:
        man = {row["image_name"]: row["split"] for row in csv.DictReader(f)}
    assert man == report.assignment

    total = 0
    for split in ("train", "val", "test"):
        with open(report.files[split], newline="", encoding="utf-8-sig") as f:
            total += sum(1 for _ in csv.DictReader(f))
    assert total == 20


def test_tolerates_misspelled_label_column(tmp_path: Path):
    columns = ["image_name", "instance_lable"]
    with open(tmp_path / "e.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for i in range(30):
            writer.writerow({"image_name": f"img_{i}.jpg", "instance_lable": f"cls{i % 3}"})

    report = random_split_csv(tmp_path / "e.csv", ratios=(0.6, 0.2, 0.2), stratify=True)
    strata = {"train": set(), "val": set(), "test": set()}
    for i in range(30):
        strata[report.assignment[f"img_{i}.jpg"]].add(i % 3)
    for seen in strata.values():
        assert seen == {0, 1, 2}


def test_missing_ts_column_raises(tmp_path: Path):
    columns = ["image_name", "instance_label"]
    with open(tmp_path / "e.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerow({"image_name": "img_0.jpg", "instance_label": "cls0"})

    with pytest.raises(ValueError, match="ts"):
        temporal_split_csv(tmp_path / "e.csv")


def test_temporal_warns_when_ts_all_missing(tmp_path: Path):
    from loguru import logger

    columns = ["image_name", "instance_label", "ts"]
    with open(tmp_path / "e.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for i in range(10):
            writer.writerow({"image_name": f"img_{i}.jpg", "instance_label": "cls0", "ts": ""})

    messages: list[str] = []
    sink = logger.add(messages.append, level="WARNING")
    try:
        temporal_split_csv(tmp_path / "e.csv", ratios=(0.5, 0.5), stratify=False)
    finally:
        logger.remove(sink)
    assert any("no-op" in m for m in messages)


def test_bad_ratios_raise(tmp_path: Path):
    csv_path = _make_csv(tmp_path / "e.csv", n=10)
    with pytest.raises(ValueError):
        random_split_csv(csv_path, ratios=(0.5, 0.2, 0.1))


def test_iso_timestamps_parse(tmp_path: Path):
    rows = [
        {"image_name": "a.jpg", "image_width": "640", "image_height": "480",
         "instance_label": "cls0", "ts": "2024-01-01T00:00:00", "camera": "cam0"},
        {"image_name": "b.jpg", "image_width": "640", "image_height": "480",
         "instance_label": "cls0", "ts": "2024-01-01T00:00:00.5", "camera": "cam0"},
        {"image_name": "c.jpg", "image_width": "640", "image_height": "480",
         "instance_label": "cls0", "ts": "2024-01-01T01:00:00", "camera": "cam0"},
    ]
    csv_path = _write_csv(tmp_path / "e.csv", rows)
    report = temporal_split_csv(csv_path, gap=1.0, ratios=(0.5, 0.5), stratify=False)
    # a and b are 0.5s apart -> same session; c is an hour later -> its own.
    assert report.assignment["a.jpg"] == report.assignment["b.jpg"]
