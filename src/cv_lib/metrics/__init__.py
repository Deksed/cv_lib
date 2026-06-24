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
