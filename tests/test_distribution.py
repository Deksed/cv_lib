"""Tests for cv_lib.viz.distribution (class-frequency bar chart)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no GUI backend during tests

import pytest

from cv_lib.viz.distribution import plot_class_distribution


def _make_labels(d: Path, class_seq: list[int]) -> Path:
    d.mkdir(parents=True)
    for i, cid in enumerate(class_seq):
        (d / f"l_{i}.txt").write_text(f"{cid} 0.5 0.5 0.2 0.2\n")
    return d


def test_single_group_bar_heights(tmp_path: Path):
    labels = _make_labels(tmp_path / "labels", [0, 0, 1, 2, 2, 2])
    fig = plot_class_distribution(labels, class_names=["a", "b", "c"])
    ax = fig.axes[0]

    assert len(ax.containers) == 1  # one group → one bar series
    heights = [p.get_height() for p in ax.containers[0]]
    assert heights == [2.0, 1.0, 3.0]


def test_multi_group_infers_num_classes(tmp_path: Path):
    train = _make_labels(tmp_path / "labels" / "train", [0, 1, 1])
    val = _make_labels(tmp_path / "labels" / "val", [0, 2])
    fig = plot_class_distribution({"train": train, "val": val})
    ax = fig.axes[0]

    assert len(ax.containers) == 2  # two groups drawn side by side
    assert ax.get_legend() is not None
    assert len(ax.containers[0]) == 3  # max class id 2 → 3 classes inferred


def test_sort_orders_by_total_descending(tmp_path: Path):
    labels = _make_labels(tmp_path / "labels", [2, 2, 2, 0, 1, 1])
    fig = plot_class_distribution(labels, class_names=["a", "b", "c"], sort=True)
    ax = fig.axes[0]

    heights = [p.get_height() for p in ax.containers[0]]
    assert heights == [3.0, 2.0, 1.0]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["c", "b", "a"]


def test_horizontal_uses_bar_widths(tmp_path: Path):
    labels = _make_labels(tmp_path / "labels", [0, 0, 1])
    fig = plot_class_distribution(labels, class_names=["a", "b"], horizontal=True)
    ax = fig.axes[0]

    widths = [p.get_width() for p in ax.containers[0]]
    assert widths == [2.0, 1.0]


def test_saves_output(tmp_path: Path):
    labels = _make_labels(tmp_path / "labels", [0, 1])
    out = tmp_path / "chart.png"
    plot_class_distribution(labels, class_names=["a", "b"], output_path=out)
    assert out.exists() and out.stat().st_size > 0


def test_empty_labels_raises(tmp_path: Path):
    (tmp_path / "labels").mkdir()
    with pytest.raises(ValueError):
        plot_class_distribution(tmp_path / "labels")
