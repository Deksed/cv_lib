"""Class-frequency bar chart for YOLO datasets.

Counts boxes per class across one or more label sets and renders a grouped bar
chart — handy for spotting class imbalance and for comparing the train/val/test
splits produced by :func:`cv_lib.data.split.train_val_test_split` side by side.

Returns a matplotlib ``Figure`` (never shows or mutates global state), so it
composes in notebooks (``fig`` auto-displays) and scripts (``fig.savefig(...)``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from cv_lib.data import class_distribution

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def _max_class_id(labels_dir: Path) -> int:
    """Highest class id present in any ``.txt`` under ``labels_dir`` (-1 if none)."""
    max_id = -1
    for label_file in labels_dir.glob("*.txt"):
        for line in label_file.read_text().splitlines():
            parts = line.split()
            if parts:
                try:
                    max_id = max(max_id, int(float(parts[0])))
                except ValueError:
                    continue
    return max_id


def _resolve_groups(labels: str | Path | Mapping[str, str | Path]) -> dict[str, Path]:
    """Normalize the ``labels`` argument into an ordered ``{group_name: dir}`` map."""
    if isinstance(labels, Mapping):
        return {str(name): Path(d) for name, d in labels.items()}
    path = Path(labels)
    return {path.name or "labels": path}


def plot_class_distribution(
    labels: str | Path | Mapping[str, str | Path],
    class_names: Sequence[str] | None = None,
    *,
    num_classes: int | None = None,
    title: str = "Class distribution",
    sort: bool = False,
    horizontal: bool = False,
    log_scale: bool = False,
    output_path: str | Path | None = None,
) -> Figure:
    """Render a grouped class-frequency bar chart.

    Args:
        labels: A single labels directory, or a mapping of group name → labels
            directory (e.g. ``{"train": "ds/labels/train", "val": ...}``) to draw
            the groups side by side.
        class_names: Class names in id order (for tick labels). Inferred as
            ``["0", "1", ...]`` when omitted.
        num_classes: Number of classes. Defaults to ``len(class_names)`` or, if
            that is also missing, ``max class id + 1`` scanned from the labels.
        title: Chart title.
        sort: Order bars by descending total frequency instead of by class id.
        horizontal: Draw horizontal bars (easier to read with many/long names).
        log_scale: Use a log scale on the count axis (for heavy imbalance).
        output_path: If given, save the figure here (``dpi=150``).

    Returns:
        The matplotlib :class:`~matplotlib.figure.Figure`.
    """
    groups = _resolve_groups(labels)
    if not groups:
        raise ValueError("No labels provided.")

    if num_classes is None:
        if class_names is not None:
            num_classes = len(class_names)
        else:
            num_classes = max((_max_class_id(d) for d in groups.values()), default=-1) + 1
    if num_classes <= 0:
        raise ValueError(
            "Could not determine the number of classes (no labels found). "
            "Pass num_classes or class_names."
        )

    counts = {name: class_distribution(d, num_classes) for name, d in groups.items()}

    names = [str(i) for i in range(num_classes)]
    if class_names is not None:
        for i, n in enumerate(class_names):
            if i < num_classes:
                names[i] = n

    order = np.arange(num_classes)
    if sort:
        totals = np.sum(list(counts.values()), axis=0)
        order = np.argsort(totals)[::-1]
    ordered_names = [names[i] for i in order]

    import matplotlib.pyplot as plt

    n_groups = len(counts)
    width = 0.8 / n_groups
    positions = np.arange(num_classes)
    span = max(6.0, num_classes * 0.6)
    figsize = (6, span) if horizontal else (span, 5)

    fig, ax = plt.subplots(figsize=figsize)
    for i, (group, values) in enumerate(counts.items()):
        offset = (i - (n_groups - 1) / 2) * width
        ordered_values = values[order]
        if horizontal:
            ax.barh(positions + offset, ordered_values, width, label=group)
        else:
            ax.bar(positions + offset, ordered_values, width, label=group)

    count_axis = ax.set_xscale if horizontal else ax.set_yscale
    if log_scale:
        count_axis("log")

    if horizontal:
        ax.set_yticks(positions)
        ax.set_yticklabels(ordered_names)
        ax.set_xlabel("Boxes")
        ax.invert_yaxis()  # first class on top
    else:
        ax.set_xticks(positions)
        ax.set_xticklabels(ordered_names, rotation=45, ha="right")
        ax.set_ylabel("Boxes")

    ax.set_title(title)
    if n_groups > 1:
        ax.legend()
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150)

    return fig
