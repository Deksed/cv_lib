"""Tests for per-class metrics and PR-curve plotting (cv_lib.metrics)."""

from __future__ import annotations

from types import SimpleNamespace

import matplotlib
import numpy as np

matplotlib.use("Agg")

from cv_lib.metrics import per_class_map, plot_pr_curves  # noqa: E402


def _fake_results() -> SimpleNamespace:
    box = SimpleNamespace(
        ap_class_index=np.array([0, 1]),
        p=np.array([0.8, 0.6]),
        r=np.array([0.7, 0.5]),
        ap50=np.array([0.75, 0.55]),
        ap=np.array([[0.75, 0.5], [0.55, 0.3]]),  # (n_classes, n_iou)
    )
    return SimpleNamespace(box=box, names={0: "car", 1: "person"})


def test_per_class_map_keys_and_values():
    out = per_class_map(_fake_results())
    assert set(out) == {"car", "person"}
    assert out["car"]["precision"] == 0.8
    assert out["car"]["recall"] == 0.7
    assert out["car"]["ap50"] == 0.75
    # ap50_95 = mean of the per-iou row [0.75, 0.5]
    assert abs(out["car"]["ap50_95"] - 0.625) < 1e-9


def test_per_class_map_respects_class_names_override():
    out = per_class_map(_fake_results(), class_names=["A", "B"])
    assert set(out) == {"A", "B"}


def test_plot_pr_curves_single_curve():
    recall = np.linspace(0, 1, 11)
    precision = 1 - recall * 0.5
    fig = plot_pr_curves(recall, precision, class_names=["car"])
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Recall"
    assert len(ax.lines) == 1


def test_plot_pr_curves_multi_class(tmp_path):
    recall = np.tile(np.linspace(0, 1, 11), (2, 1))
    precision = np.vstack([1 - recall[0] * 0.3, 1 - recall[1] * 0.6])
    out = tmp_path / "pr.png"
    fig = plot_pr_curves(recall, precision, class_names=["car", "person"], output_path=str(out))
    assert len(fig.axes[0].lines) == 2
    assert out.exists()
