"""Tests for cv_lib.metrics (confusion matrix figure + mAP summary).

`summarize_map` is exercised against a lightweight stub that mimics the bits of
an Ultralytics Results object it reads (`.names`, `.box.map50/.map/.ap/...`).
"""

from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless: never pop a window during tests

from cv_lib.metrics import plot_confusion_matrix, summarize_map


class _Box:
    def __init__(self, map50, map95, ap_class_index, ap):
        self.map50 = map50
        self.map = map95
        self.ap_class_index = ap_class_index
        self.ap = ap


class _Results:
    def __init__(self, names, box):
        self.names = names
        self.box = box


def test_summarize_map_overall_and_per_class():
    box = _Box(
        map50=0.82,
        map95=0.61,
        ap_class_index=np.array([0, 1]),
        # ap is (num_classes, num_iou_thresholds); column 0 is AP@0.5
        ap=np.array([[0.9, 0.7], [0.5, 0.3]]),
    )
    results = _Results(names={0: "car", 1: "person"}, box=box)

    summary = summarize_map(results)

    assert summary["mAP50"] == 0.82
    assert summary["mAP50-95"] == 0.61
    assert summary["AP50/car"] == 0.9
    assert summary["AP50/person"] == 0.5


def test_summarize_map_respects_explicit_class_names():
    box = _Box(0.5, 0.4, np.array([0]), np.array([[0.6, 0.4]]))
    results = _Results(names={0: "ignored"}, box=box)

    summary = summarize_map(results, class_names=["override"])
    assert "AP50/override" in summary


def test_summarize_map_handles_missing_ap():
    box = _Box(0.3, 0.2, ap_class_index=np.array([]), ap=None)
    results = _Results(names={0: "car"}, box=box)

    summary = summarize_map(results)
    assert summary == {"mAP50": 0.3, "mAP50-95": 0.2}


def test_plot_confusion_matrix_returns_figure():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    fig = plot_confusion_matrix(y_true, y_pred, class_names=["a", "b", "c"])

    from matplotlib.figure import Figure

    assert isinstance(fig, Figure)
    assert fig.axes  # at least one axis was drawn
    matplotlib.pyplot.close(fig)
