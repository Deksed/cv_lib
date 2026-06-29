"""Pick a deployment confidence threshold by sweeping ``model.val()``.

Validation mAP is computed at a near-zero confidence so the PR curve is
complete, but a *deployed* model needs a single operating point. This sweeps a
range of confidence thresholds, records precision/recall at each, and reports
the threshold that maximises the chosen metric (F1 by default) — so the cutoff
is chosen from data instead of by eye.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def best_operating_point(
    thresholds: list[float],
    precision: list[float],
    recall: list[float],
    metric: str = "f1",
) -> dict[str, float]:
    """Return the threshold maximising ``metric`` over parallel P/R arrays.

    Args:
        thresholds: Confidence thresholds (parallel to ``precision``/``recall``).
        precision:  Precision at each threshold.
        recall:     Recall at each threshold.
        metric:     ``"f1"`` (default), ``"precision"`` or ``"recall"``.

    Returns:
        ``{"threshold", "precision", "recall", "f1", "score"}`` for the best point.
    """
    if not (len(thresholds) == len(precision) == len(recall)) or not thresholds:
        raise ValueError("thresholds, precision and recall must be non-empty and equal length.")

    best: dict[str, float] | None = None
    for t, p, r in zip(thresholds, precision, recall):
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        score = {"f1": f1, "precision": p, "recall": r}[metric]
        if best is None or score > best["score"]:
            best = {"threshold": t, "precision": p, "recall": r, "f1": f1, "score": score}
    return best


@dataclass
class ThresholdReport:
    """Result of :func:`sweep_threshold`."""

    metric: str
    best: dict[str, float]
    thresholds: list[float] = field(default_factory=list)
    precision: list[float] = field(default_factory=list)
    recall: list[float] = field(default_factory=list)
    f1: list[float] = field(default_factory=list)

    def print(self) -> None:
        b = self.best
        print(f"\nThreshold sweep (optimise {self.metric}):")
        print(f"  {'conf':>6}  {'P':>7}  {'R':>7}  {'F1':>7}")
        for t, p, r, f in zip(self.thresholds, self.precision, self.recall, self.f1):
            mark = "  <-- best" if abs(t - b["threshold"]) < 1e-9 else ""
            print(f"  {t:>6.2f}  {p:>7.4f}  {r:>7.4f}  {f:>7.4f}{mark}")
        print(
            f"\nBest conf={b['threshold']:.2f}  "
            f"P={b['precision']:.4f}  R={b['recall']:.4f}  F1={b['f1']:.4f}"
        )


def sweep_threshold(
    model,
    data: str,
    *,
    thresholds: list[float] | None = None,
    iou: float = 0.6,
    split: str = "val",
    metric: str = "f1",
    device: str | None = None,
) -> ThresholdReport:
    """Run ``model.val()`` across confidence thresholds and find the best point.

    Args:
        model:      a loaded Ultralytics ``YOLO`` (or any object with ``.val``).
        data:       path to a YOLO ``data.yaml``.
        thresholds: confidence values to test (default ``0.1..0.9`` step 0.1).
        iou:        NMS IoU threshold for validation.
        split:      dataset split to evaluate.
        metric:     metric to maximise (see :func:`best_operating_point`).
        device:     optional device override.

    Returns:
        A :class:`ThresholdReport`.
    """
    if thresholds is None:
        thresholds = [round(0.1 * i, 2) for i in range(1, 10)]

    ps: list[float] = []
    rs: list[float] = []
    for conf in thresholds:
        kwargs: dict = {"data": data, "conf": conf, "iou": iou, "split": split, "verbose": False}
        if device is not None:
            kwargs["device"] = device
        results = model.val(**kwargs)
        ps.append(float(results.box.mp))  # mean precision over classes
        rs.append(float(results.box.mr))  # mean recall over classes

    f1s = [2 * p * r / (p + r) if (p + r) > 0 else 0.0 for p, r in zip(ps, rs)]
    best = best_operating_point(thresholds, ps, rs, metric)
    return ThresholdReport(
        metric=metric, best=best, thresholds=thresholds, precision=ps, recall=rs, f1=f1s
    )
