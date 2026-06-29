"""Evaluation helpers: confusion matrix display, per-class mAP summary."""

from __future__ import annotations

import numpy as np


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    normalize: bool = True,
    title: str = "Confusion Matrix",
):
    """
    Display a confusion matrix using sklearn + matplotlib.

    Args:
        y_true:       ground truth class indices (1-D int array)
        y_pred:       predicted class indices (1-D int array)
        class_names:  list of class name strings
        normalize:    show proportions instead of raw counts
        title:        plot title

    Returns:
        matplotlib Figure
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    norm = "true" if normalize else None
    cm = confusion_matrix(y_true, y_pred, normalize=norm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(max(6, len(class_names)), max(5, len(class_names))))
    disp.plot(ax=ax, colorbar=True, xticks_rotation="vertical")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def summarize_map(results, class_names: list[str] | None = None) -> dict[str, float]:
    """
    Extract per-class AP and overall mAP50 / mAP50-95 from Ultralytics Results.

    Args:
        results:      return value of model.val()
        class_names:  if None, uses results.names

    Returns:
        dict with keys: 'mAP50', 'mAP50-95', and per-class 'AP50/<name>'
    """
    names = class_names or list(results.names.values())
    summary: dict[str, float] = {
        "mAP50": float(results.box.map50),
        "mAP50-95": float(results.box.map),
    }
    if hasattr(results.box, "ap_class_index") and results.box.ap is not None:
        for i, ap_val in zip(results.box.ap_class_index, results.box.ap):
            name = names[i] if i < len(names) else str(i)
            summary[f"AP50/{name}"] = float(ap_val[0]) if ap_val.ndim else float(ap_val)
    return summary


def per_class_map(results, class_names: list[str] | None = None) -> dict[str, dict[str, float]]:
    """Break Ultralytics ``model.val()`` results down per class.

    Args:
        results:      return value of ``model.val()``
        class_names:  override for ``results.names``

    Returns:
        ``{class_name: {"precision", "recall", "ap50", "ap50_95"}}`` for every
        class that appears in ``results.box.ap_class_index``.
    """
    names = class_names or list(results.names.values())
    box = results.box
    out: dict[str, dict[str, float]] = {}

    def _col(attr: str):
        return getattr(box, attr, None)

    p, r = _col("p"), _col("r")
    ap50, ap = _col("ap50"), _col("ap")
    class_index = getattr(box, "ap_class_index", None)
    if class_index is None:
        return out
    for pos, cls_idx in enumerate(class_index):
        name = names[cls_idx] if cls_idx < len(names) else str(cls_idx)
        ap_val = ap[pos] if ap is not None else None
        out[name] = {
            "precision": float(p[pos]) if p is not None else 0.0,
            "recall": float(r[pos]) if r is not None else 0.0,
            "ap50": float(ap50[pos]) if ap50 is not None else 0.0,
            "ap50_95": float(ap_val.mean() if hasattr(ap_val, "mean") else ap_val)
            if ap_val is not None
            else 0.0,
        }
    return out


def plot_pr_curves(
    recall: np.ndarray,
    precision: np.ndarray,
    class_names: list[str] | None = None,
    title: str = "Precision-Recall",
    output_path: str | None = None,
):
    """Plot one precision-recall curve per class.

    Args:
        recall:       recall sample points — 1-D ``(n_points,)`` for a single
                      curve, or 2-D ``(n_classes, n_points)`` for several.
        precision:    precision values, same shape as ``recall``.
        class_names:  legend labels (one per row when 2-D).
        title:        plot title.
        output_path:  if given, save the figure there.

    Returns:
        matplotlib Figure
    """
    import matplotlib.pyplot as plt

    recall = np.asarray(recall)
    precision = np.asarray(precision)
    if recall.ndim == 1:
        recall = recall[None, :]
        precision = precision[None, :]

    fig, ax = plt.subplots(figsize=(7, 6))
    for i in range(recall.shape[0]):
        label = class_names[i] if class_names and i < len(class_names) else f"class {i}"
        ax.plot(recall[i], precision[i], label=label)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.legend(loc="lower left", fontsize="small")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig
