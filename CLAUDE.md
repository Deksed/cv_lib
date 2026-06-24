# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

CV utility library for iterating on object detection models. Core stack: **Ultralytics (YOLO)** as primary framework, PyTorch + torchvision for custom components, OpenCV for image I/O and visualization, Jupyter notebooks for exploration.

```
src/cv_lib/
├── viz/
│   ├── __init__.py          # (empty — re-exports TBD)
│   └── compare.py           # side-by-side GT vs prediction visualizer
├── data/
│   └── __init__.py          # YOLO format parsing, class distribution, iter pairs
├── metrics/
│   └── __init__.py          # confusion matrix, mAP summary
└── export.py                # ONNX / TensorRT export + validate
scripts/
└── compare_gt_pred.py       # CLI wrapper over viz.compare
tests/
└── conftest.py              # pytest fixtures: sample_image, yolo_label_file
configs/                     # YOLO data.yaml templates (currently empty)
notebooks/                   # Jupyter experiments (currently empty)
```

## Setup

Целевая машина: Python 3.12, CUDA ≥ 11.8 → cu118-сборка torch (2.4–2.5).

### uv (рекомендуется)

```bash
uv pip install -e ".[cu118,dev]"
```

`[tool.uv.sources]` в `pyproject.toml` автоматически берёт `torch`/`torchvision`
с `https://download.pytorch.org/whl/cu118` — дополнительных флагов не нужно.

### pip

```bash
pip install -r requirements-torch.txt   # torch + torchvision с cu118-индекса
pip install -e ".[dev]"
```

Для CUDA 12.4+ / Python 3.13 — требуется cu124-сборка torch (установить вручную).
В `pyproject.toml` убрать ограничение `<3.13` из `requires-python`.

Lint + format: `ruff check src/ scripts/` / `ruff format src/ scripts/`  
Tests: `pytest`

## Data & Annotations

Dataset root and annotation paths are **never hardcoded**. Always read from:
- `DATA_ROOT` env var or a config parameter for the dataset directory
- `CVAT_ANNOTATIONS_PATH` env var or config for CVAT export paths

Datasets follow **YOLO format** (`.yaml` config + `images/` + `labels/` structure). CVAT exports are typically in YOLO 1.1 or COCO JSON — check the format before parsing.

Example pattern:
```python
import os
data_root = os.environ["DATA_ROOT"]  # or cfg.data_root
```

## Device

Always abstract the device — never use raw `.cuda()`:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
tensor = tensor.to(device)
```

## Reproducibility

Training scripts must set seeds explicitly:
```python
import random, numpy as np, torch
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

## Ultralytics Conventions

- Model configs live in `.yaml` files (architecture + dataset path)
- Use `model.train(data="path/to/data.yaml", ...)` — pass the yaml, not raw paths
- `results.boxes`, `results.masks` — use the Ultralytics Results API instead of raw tensor indexing
- For custom training: subclass `ultralytics.data.Dataset` or use `data` param with a prepared yaml

## Modules

### cv_lib.viz.compare

Existing functions:

```python
compare_gt_pred(image_path, model_path, class_names,
                conf_threshold=0.25, label_path=None,
                output_path=None, show=True) -> np.ndarray
```
Side-by-side GT (left) vs predictions (right). Auto-detects Jupyter vs terminal:
notebook → `matplotlib`, terminal → `cv2.imshow`.
Label auto-resolve: same dir first, then `images/ → labels/` swap.

```python
load_yolo_gt(label_path, img_w, img_h) -> tuple[np.ndarray, np.ndarray]
# boxes_xyxy (N,4) float32, class_ids (N,) int32
```

Internal helpers (don't expose in public API): `_draw_boxes`, `_resolve_label_path`, `_add_panel_label`, `_in_notebook`.

New viz modules to add: `viz/batch.py` (make_grid), `viz/distribution.py` (class bar chart), `viz/errors.py` (FP/FN analysis).

### cv_lib.data

```python
load_dataset_yaml(path) -> dict
class_names_from_yaml(path) -> list[str]
iter_image_label_pairs(images_dir, labels_dir=None, extensions=...) -> list[tuple[Path, Path]]
class_distribution(labels_dir, num_classes) -> np.ndarray  # shape (num_classes,)
data_root() -> Path  # reads DATA_ROOT env var, raises EnvironmentError if unset
```

`iter_image_label_pairs` auto-infers `labels_dir` by swapping `images` in the path.
`label_path` in the returned pair may not exist (image has no annotation).

To add: `data/inspect.py` (dataset health check, missing labels, corrupt images, bbox sanity), `data/convert.py` (CVAT XML / COCO JSON → YOLO txt).

### cv_lib.metrics

```python
plot_confusion_matrix(y_true, y_pred, class_names, normalize=True) -> Figure
summarize_map(results, class_names=None) -> dict[str, float]
# keys: "mAP50", "mAP50-95", "AP50/<name>"
```

`summarize_map` reads `results.box.map50`, `results.box.map`, `results.box.ap` from Ultralytics `model.val()` return value.

### cv_lib.export

```python
export_onnx(model, path, input_shape=(1,3,640,640), dynamic=True, simplify=True) -> Path
export_trt(onnx_path, engine_path, fp16=True, workspace_gb=4) -> Path  # requires tensorrt
validate_export(pytorch_output, exported_output, atol=1e-4) -> None   # raises ValueError on mismatch
```

`export_onnx` moves the Ultralytics-generated `.onnx` to `path` after export.
`tensorrt` is not in `[project.dependencies]` — must be installed separately.

## Visualization Conventions

- New viz functions go in `src/cv_lib/viz/` as separate modules
- Draw on a **copy** of the image — never mutate the input (`img.copy()` first)
- Return `np.ndarray` (BGR, uint8) from all viz functions — OpenCV-compatible
- Accept both `np.ndarray` (H×W×C) and `torch.Tensor` (C×H×W float) — normalize internally
- Batch grid: `torchvision.utils.make_grid` → `.permute(1,2,0).numpy()` → `plt.imshow`
- Confusion matrix: `sklearn.metrics.ConfusionMatrixDisplay`

## Export

`src/cv_lib/export.py` exists. Use `export_onnx` / `export_trt` / `validate_export` from there — don't call Ultralytics export API directly in scripts.

- ONNX: dynamic axes for variable batch size; onnx-simplifier runs by default
- TensorRT: FP16 enabled if `builder.platform_has_fast_fp16`; `workspace_gb=4` by default
- Always call `validate_export(pytorch_out, exported_out, atol=1e-4)` after export

## Code Style

- Type hints on all public functions
- Return `np.ndarray` from viz functions (OpenCV-compatible), not tensors
- Avoid in-place tensor ops on inputs — clone/copy first
- Ruff rules: `E`, `F`, `I`, `UP`; line length 100; `E501` ignored (warning only)
- Notebooks exempt from `E402` / `F401`

## Roadmap — What's Missing

Ordered by priority for the iteration loop:

1. **`scripts/eval.py`** — run `model.val(data=yaml)`, print mAP table via `summarize_map`, save confusion matrix to file
2. **`scripts/batch_infer.py`** — run model over a directory, save predictions as YOLO labels or visualized images
3. **`data/inspect.py`** — find missing labels, corrupt images, out-of-bound boxes, print class imbalance report
4. **`viz/batch.py`** — `show_batch(images, labels, class_names)` using `make_grid`; accepts both `np.ndarray` and `torch.Tensor`
5. **`data/convert.py`** — CVAT XML / COCO JSON → YOLO `.txt` converter
6. **`viz/errors.py`** — find and render false positives, false negatives, worst-confidence predictions
7. **`train/__init__.py`** — thin wrapper over `model.train()` with seed setup and config save

Don't add W&B/MLflow — Ultralytics writes W&B natively. Don't write a custom Dataset subclass until non-standard augmentation is required.
