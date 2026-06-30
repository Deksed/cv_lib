"""Tests for the top-level cv_lib public API (lazy re-exports, __all__)."""

from __future__ import annotations

import cv_lib

# The public API contract (issue #1). Keep in sync with cv_lib._EXPORTS.
EXPECTED_EXPORTS = {
    "compare_gt_pred",
    "load_yolo_gt",
    "show_batch",
    "find_errors",
    "render_errors",
    "ErrorEntry",
    "plot_class_distribution",
    "augment_preview",
    "default_transform",
    "load_dataset_yaml",
    "class_names_from_yaml",
    "iter_image_label_pairs",
    "class_distribution",
    "data_root",
    "inspect_dataset",
    "InspectReport",
    "cvat_xml_to_yolo",
    "coco_json_to_yolo",
    "cvat_csv_to_yolo",
    "cvat_csv_gt",
    "predictions_to_cvat_csv",
    "query_cvat_csv",
    "build_pipeline",
    "generate_dvc_yaml",
    "generate_params_yaml",
    "PipelineConfig",
    "yolo_to_coco",
    "yolo_to_voc",
    "yolo_to_cvat_csv",
    "voc_to_yolo",
    "train_val_test_split",
    "SplitReport",
    "merge_datasets",
    "MergeReport",
    "DatasetSource",
    "source_from_root",
    "repair_labels",
    "RepairReport",
    "remap_labels",
    "RemapReport",
    "audit_labels",
    "QAReport",
    "find_duplicates",
    "check_split_leakage",
    "extract_crops",
    "CropReport",
    "autolabel",
    "rank_for_labeling",
    "uncertainty_score",
    "sliced_predict",
    "generate_tiles",
    "plot_confusion_matrix",
    "summarize_map",
    "per_class_map",
    "plot_pr_curves",
    "sweep_threshold",
    "best_operating_point",
    "set_seeds",
    "train",
    "export_onnx",
    "export_trt",
    "validate_export",
}


def test_version_is_exposed():
    assert isinstance(cv_lib.__version__, str)
    assert "__version__" in cv_lib.__all__


def test_all_lists_exactly_the_public_api():
    assert set(cv_lib.__all__) == EXPECTED_EXPORTS | {"__version__"}


def test_dir_matches_all():
    assert set(dir(cv_lib)) == set(cv_lib.__all__)


def test_lazy_access_resolves_callables():
    # Touch a few names that don't require heavy deps (torch/ultralytics)
    for name in ("data_root", "cvat_csv_to_yolo", "query_cvat_csv", "summarize_map"):
        attr = getattr(cv_lib, name)
        assert callable(attr)
        assert attr.__name__ == name


def test_lazy_access_caches_in_globals():
    # First access goes through __getattr__; afterwards it lives in the module dict.
    _ = cv_lib.coco_json_to_yolo
    assert "coco_json_to_yolo" in vars(cv_lib)


def test_unknown_attribute_raises():
    import pytest

    with pytest.raises(AttributeError):
        _ = cv_lib.definitely_not_a_real_export
