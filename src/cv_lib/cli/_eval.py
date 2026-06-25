"""`cvlib eval` — run model.val() and print a mAP table + confusion matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from cv_lib.cli._common import add_verbose

HELP = "Evaluate a YOLO model: mAP table + confusion matrix PNG."


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="Path to Ultralytics .pt model.")
    parser.add_argument("--data", required=True, help="Path to YOLO data.yaml.")
    parser.add_argument(
        "--conf",
        type=float,
        default=0.001,
        help="Confidence threshold for val (default: 0.001 — standard for mAP).",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.6,
        help="IoU threshold for NMS during val (default: 0.6).",
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=["val", "test", "train"],
        help="Dataset split to evaluate on (default: val).",
    )
    parser.add_argument(
        "--cm-out",
        default=None,
        metavar="PATH",
        help="Where to save confusion matrix PNG (default: <data_yaml_dir>/confusion_matrix.png).",
    )
    parser.add_argument("--no-cm", action="store_true", help="Skip saving the confusion matrix.")
    parser.add_argument(
        "--device",
        default=None,
        help="Device override, e.g. 'cpu', '0', 'cuda:0' (default: auto).",
    )
    add_verbose(parser)


def run(args: argparse.Namespace) -> None:
    from ultralytics import YOLO

    from cv_lib.metrics import plot_confusion_matrix, summarize_map

    model = YOLO(args.model)

    val_kwargs: dict = {
        "data": args.data,
        "conf": args.conf,
        "iou": args.iou,
        "split": args.split,
        "verbose": args.verbose,
    }
    if args.device is not None:
        val_kwargs["device"] = args.device

    logger.info("Evaluating {} on {} [{}] …", args.model, args.data, args.split)
    results = model.val(**val_kwargs)

    summary = summarize_map(results)

    # --- mAP table ---
    col_w = max(len(k) for k in summary) + 2
    print("\n" + "─" * (col_w + 12))
    print(f"{'Metric':<{col_w}}  {'Value':>8}")
    print("─" * (col_w + 12))
    top_keys = ["mAP50", "mAP50-95"]
    for k in top_keys:
        if k in summary:
            print(f"{k:<{col_w}}  {summary[k]:>8.4f}")
    print("─" * (col_w + 12))
    for k, v in summary.items():
        if k not in top_keys:
            print(f"{k:<{col_w}}  {v:>8.4f}")
    print("─" * (col_w + 12))

    # --- confusion matrix ---
    if not args.no_cm:
        cm_path = (
            Path(args.cm_out)
            if args.cm_out
            else Path(args.data).parent / "confusion_matrix.png"
        )
        cm_path.parent.mkdir(parents=True, exist_ok=True)

        class_names = list(results.names.values())

        # Ultralytics stores raw confusion matrix in results.confusion_matrix.matrix (numpy)
        cm_matrix = None
        if hasattr(results, "confusion_matrix") and results.confusion_matrix is not None:
            cm_matrix = results.confusion_matrix.matrix

        if cm_matrix is not None:
            import numpy as np

            # matrix shape: (num_classes+1, num_classes+1) incl. background row/col
            n = len(class_names)
            cm_classes = cm_matrix[:n, :n].astype(int)
            y_true_flat: list[int] = []
            y_pred_flat: list[int] = []
            for true_idx in range(n):
                for pred_idx in range(n):
                    count = int(cm_classes[true_idx, pred_idx])
                    y_true_flat.extend([true_idx] * count)
                    y_pred_flat.extend([pred_idx] * count)

            if y_true_flat:
                fig = plot_confusion_matrix(
                    np.array(y_true_flat),
                    np.array(y_pred_flat),
                    class_names=class_names,
                )
                fig.savefig(cm_path, dpi=150, bbox_inches="tight")
                logger.info("Confusion matrix saved → {}", cm_path)
            else:
                logger.warning("Confusion matrix is empty, skipping save.")
        else:
            logger.warning("Confusion matrix not available in results, skipping save.")
