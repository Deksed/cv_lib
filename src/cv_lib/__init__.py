"""cv_lib — CV utility library for object detection model iteration.

Public API. Names are re-exported lazily (PEP 562) so that ``import cv_lib``
stays cheap and only pulls heavy dependencies (OpenCV, torch, Ultralytics)
when a specific helper is first accessed.

    >>> import cv_lib
    >>> cv_lib.__version__
    '0.2.0'
    >>> from cv_lib import inspect_dataset, cvat_csv_to_yolo, summarize_map

Anything not listed in ``__all__`` is internal and may change without notice.
Submodules (``cv_lib.viz``, ``cv_lib.data``, ``cv_lib.metrics``,
``cv_lib.train``, ``cv_lib.export``, ``cv_lib.cli``) remain importable directly.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

__version__ = "0.2.0"

# Public name → submodule it lives in. Drives both __all__ and lazy __getattr__.
_EXPORTS: dict[str, str] = {
    # viz
    "compare_gt_pred": "cv_lib.viz",
    "load_yolo_gt": "cv_lib.viz",
    "show_batch": "cv_lib.viz",
    "find_errors": "cv_lib.viz",
    "render_errors": "cv_lib.viz",
    "ErrorEntry": "cv_lib.viz",
    "plot_class_distribution": "cv_lib.viz",
    "augment_preview": "cv_lib.viz",
    "default_transform": "cv_lib.viz",
    # data
    "load_dataset_yaml": "cv_lib.data",
    "class_names_from_yaml": "cv_lib.data",
    "iter_image_label_pairs": "cv_lib.data",
    "class_distribution": "cv_lib.data",
    "data_root": "cv_lib.data",
    # data.inspect
    "inspect_dataset": "cv_lib.data.inspect",
    "InspectReport": "cv_lib.data.inspect",
    # data.convert
    "cvat_xml_to_yolo": "cv_lib.data.convert",
    "coco_json_to_yolo": "cv_lib.data.convert",
    "cvat_csv_to_yolo": "cv_lib.data.convert",
    "cvat_csv_gt": "cv_lib.data.convert",
    "predictions_to_cvat_csv": "cv_lib.data.convert",
    "query_cvat_csv": "cv_lib.data.convert",
    # data.dvc_gen
    "build_pipeline": "cv_lib.data.dvc_gen",
    "generate_dvc_yaml": "cv_lib.data.dvc_gen",
    "generate_params_yaml": "cv_lib.data.dvc_gen",
    "PipelineConfig": "cv_lib.data.dvc_gen",
    "yolo_to_coco": "cv_lib.data.convert",
    "yolo_to_voc": "cv_lib.data.convert",
    "yolo_to_cvat_csv": "cv_lib.data.convert",
    "voc_to_yolo": "cv_lib.data.convert",
    # data.split
    "train_val_test_split": "cv_lib.data.split",
    "SplitReport": "cv_lib.data.split",
    # data.csv_split
    "random_split_csv": "cv_lib.data.csv_split",
    "temporal_split_csv": "cv_lib.data.csv_split",
    "camera_temporal_split_csv": "cv_lib.data.csv_split",
    "CsvSplitReport": "cv_lib.data.csv_split",
    # data.merge
    "merge_datasets": "cv_lib.data.merge",
    "MergeReport": "cv_lib.data.merge",
    "DatasetSource": "cv_lib.data.merge",
    "source_from_root": "cv_lib.data.merge",
    # data.repair
    "repair_labels": "cv_lib.data.repair",
    "RepairReport": "cv_lib.data.repair",
    # data.remap
    "remap_labels": "cv_lib.data.remap",
    "RemapReport": "cv_lib.data.remap",
    # data.qa
    "audit_labels": "cv_lib.data.qa",
    "QAReport": "cv_lib.data.qa",
    # data.dedup
    "find_duplicates": "cv_lib.data.dedup",
    "check_split_leakage": "cv_lib.data.dedup",
    # data.crops
    "extract_crops": "cv_lib.data.crops",
    "CropReport": "cv_lib.data.crops",
    # data.autolabel / mining
    "autolabel": "cv_lib.data.autolabel",
    "rank_for_labeling": "cv_lib.data.mining",
    "uncertainty_score": "cv_lib.data.mining",
    # infer
    "sliced_predict": "cv_lib.infer.tiled",
    "generate_tiles": "cv_lib.infer.tiled",
    # metrics
    "plot_confusion_matrix": "cv_lib.metrics",
    "summarize_map": "cv_lib.metrics",
    "per_class_map": "cv_lib.metrics",
    "plot_pr_curves": "cv_lib.metrics",
    "sweep_threshold": "cv_lib.metrics.threshold",
    "best_operating_point": "cv_lib.metrics.threshold",
    # train
    "set_seeds": "cv_lib.train",
    "train": "cv_lib.train",
    # export
    "export_onnx": "cv_lib.export",
    "export_trt": "cv_lib.export",
    "validate_export": "cv_lib.export",
}

__all__ = ["__version__", *sorted(_EXPORTS)]


def __getattr__(name: str):
    """Lazily import a public name from its submodule on first access."""
    module_path = _EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    attr = getattr(importlib.import_module(module_path), name)
    globals()[name] = attr  # cache so subsequent lookups skip __getattr__
    return attr


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:
    # Make the lazy names visible to type checkers / IDEs. These names are
    # exported lazily via __getattr__; __all__ is built from _EXPORTS, so ruff
    # cannot see them as "used" here — hence the per-line F401 suppressions.
    from cv_lib.data import (  # noqa: F401
        class_distribution,
        class_names_from_yaml,
        data_root,
        iter_image_label_pairs,
        load_dataset_yaml,
    )
    from cv_lib.data.autolabel import autolabel  # noqa: F401
    from cv_lib.data.convert import (  # noqa: F401
        coco_json_to_yolo,
        cvat_csv_gt,
        cvat_csv_to_yolo,
        cvat_xml_to_yolo,
        predictions_to_cvat_csv,
        query_cvat_csv,
        voc_to_yolo,
        yolo_to_coco,
        yolo_to_cvat_csv,
        yolo_to_voc,
    )
    from cv_lib.data.crops import CropReport, extract_crops  # noqa: F401
    from cv_lib.data.csv_split import (  # noqa: F401
        CsvSplitReport,
        camera_temporal_split_csv,
        random_split_csv,
        temporal_split_csv,
    )
    from cv_lib.data.dedup import check_split_leakage, find_duplicates  # noqa: F401
    from cv_lib.data.dvc_gen import (  # noqa: F401
        PipelineConfig,
        build_pipeline,
        generate_dvc_yaml,
        generate_params_yaml,
    )
    from cv_lib.data.inspect import InspectReport, inspect_dataset  # noqa: F401
    from cv_lib.data.merge import (  # noqa: F401
        DatasetSource,
        MergeReport,
        merge_datasets,
        source_from_root,
    )
    from cv_lib.data.mining import rank_for_labeling, uncertainty_score  # noqa: F401
    from cv_lib.data.qa import QAReport, audit_labels  # noqa: F401
    from cv_lib.data.remap import RemapReport, remap_labels  # noqa: F401
    from cv_lib.data.repair import RepairReport, repair_labels  # noqa: F401
    from cv_lib.data.split import SplitReport, train_val_test_split  # noqa: F401
    from cv_lib.export import export_onnx, export_trt, validate_export  # noqa: F401
    from cv_lib.infer.tiled import generate_tiles, sliced_predict  # noqa: F401
    from cv_lib.metrics import (  # noqa: F401
        per_class_map,
        plot_confusion_matrix,
        plot_pr_curves,
        summarize_map,
    )
    from cv_lib.metrics.threshold import best_operating_point, sweep_threshold  # noqa: F401
    from cv_lib.train import set_seeds, train  # noqa: F401
    from cv_lib.viz import (  # noqa: F401
        ErrorEntry,
        augment_preview,
        compare_gt_pred,
        default_transform,
        find_errors,
        load_yolo_gt,
        plot_class_distribution,
        render_errors,
        show_batch,
    )
