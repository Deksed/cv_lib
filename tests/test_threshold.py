"""Tests for cv_lib.metrics.threshold (confidence operating-point sweep)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cv_lib.metrics.threshold import best_operating_point, sweep_threshold


def test_best_operating_point_f1():
    thresholds = [0.1, 0.3, 0.5, 0.7]
    precision = [0.5, 0.7, 0.9, 0.95]
    recall = [0.95, 0.9, 0.7, 0.3]
    best = best_operating_point(thresholds, precision, recall, "f1")
    # F1 is highest where P and R are balanced (0.3 or 0.5); check it's a real max
    assert best["threshold"] in (0.3, 0.5)
    assert best["f1"] == max(
        2 * p * r / (p + r) for p, r in zip(precision, recall)
    )


def test_best_operating_point_precision_metric():
    best = best_operating_point([0.1, 0.9], [0.5, 0.99], [0.9, 0.2], "precision")
    assert best["threshold"] == 0.9
    assert best["precision"] == 0.99


def test_best_operating_point_validates_lengths():
    with pytest.raises(ValueError):
        best_operating_point([0.1], [0.5, 0.6], [0.9], "f1")


def test_sweep_threshold_with_fake_model():
    # Fake model whose precision rises and recall falls with conf.
    def make_results(conf):
        mp = min(0.99, 0.5 + conf)
        mr = max(0.0, 1.0 - conf)
        return SimpleNamespace(box=SimpleNamespace(mp=mp, mr=mr))

    class FakeModel:
        def val(self, **kwargs):
            return make_results(kwargs["conf"])

    report = sweep_threshold(FakeModel(), "data.yaml", thresholds=[0.2, 0.4, 0.6])
    assert report.thresholds == [0.2, 0.4, 0.6]
    assert len(report.f1) == 3
    assert report.best["threshold"] in (0.2, 0.4, 0.6)
