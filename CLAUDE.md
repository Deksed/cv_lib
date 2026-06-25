# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

CV utility library for iterating on object detection models. Core stack: **Ultralytics (YOLO)** as primary framework, PyTorch + torchvision for custom components, OpenCV for image I/O and visualization, Jupyter notebooks for exploration.

```
src/cv_lib/
├── cli/                     # cvlib unified CLI (entry point `cvlib` in pyproject)
│   ├── __init__.py          # build_parser() + main() dispatch; COMMANDS registry
│   ├── _common.py           # configure_console, load_env, resolve_names, apply_data_root
│   └── _eval/_infer/_compare/_inspect/_convert/_bench.py  # add_arguments(p) + run(args)
├── viz/
│   ├── __init__.py          # re-exports: compare_gt_pred, load_yolo_gt, show_batch, find_errors, render_errors
│   ├── compare.py           # side-by-side GT vs prediction visualizer
│   ├── batch.py             # show_batch() — image grid with YOLO box overlays
│   └── errors.py            # find_errors() / render_errors() — FP/FN analysis
├── data/
│   ├── __init__.py          # YOLO format parsing, class distribution, iter pairs
│   ├── inspect.py           # dataset health check: corrupt images, missing labels, OOB boxes
│   └── convert.py           # CVAT XML / COCO JSON → YOLO txt
├── metrics/
│   └── __init__.py          # confusion matrix, mAP summary
├── train/
│   └── __init__.py          # train() wrapper: seeds + config snapshot + model.train()
└── export.py                # ONNX / TensorRT export + validate
scripts/                     # thin shims over cv_lib.cli (backwards compat; prefer `cvlib`)
├── eval.py                  # ≡ cvlib eval   — model.val() → mAP table + confusion matrix
├── batch_infer.py           # ≡ cvlib infer  — batch inference → YOLO labels / annotated imgs
├── compare_gt_pred.py       # ≡ cvlib compare — GT vs prediction side-by-side
├── check_infer.py           # ≡ cvlib bench  — inference sanity check + latency benchmark
└── compare_runs.py          # ≡ cvlib compare-runs — train run configs + metrics table
tests/
├── conftest.py              # pytest fixtures: sample_image, yolo_label_file
├── test_cli.py
├── test_data_inspect.py
├── test_data_convert.py
├── test_viz_batch.py
└── test_viz_errors.py
configs/                     # YOLO data.yaml templates (currently empty)
notebooks/                   # Jupyter experiments
.env.example                 # env var template — copy to .env
```

## CLI

Единая точка входа `cvlib` (`[project.scripts]` → `cv_lib.cli:main`):
`cvlib inspect|convert|compare|infer|eval|bench|compare-runs`. Реализация команд —
в `cv_lib.cli._<cmd>` (каждый модуль = `HELP` + `add_arguments(parser)` + `run(args)`,
опц. `EPILOG`), зарегистрированы в `COMMANDS`.

Добавление новой подкоманды: создать `cv_lib/cli/_<name>.py` с `HELP`/`add_arguments`/`run`
и вписать в `COMMANDS` в `cv_lib/cli/__init__.py`. Скрипты в `scripts/` — тонкие шимы, дублировать
в них логику нельзя. UTF-8 для вывода обеспечивает `configure_console()` (рамки `─`/`█` ломают cp1251).

Логирование: `add_verbose(parser)` добавляет общий `--verbose`, `setup_logging(verbose)`
настраивает loguru (DEBUG/INFO → stderr). Статус-сообщения — через `logger`, форматированные
таблицы остаются на `print` (stdout). `main()` и шимы вызывают `setup_logging` после парсинга.

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
Tests: `uv run --extra dev pytest`

## Data & Annotations

Dataset root and annotation paths are **never hardcoded**. Always read from:
- `DATA_ROOT` env var or a config parameter for the dataset directory
- `CVAT_ANNOTATIONS_PATH` env var or config for CVAT export paths

`.env` is loaded automatically by all scripts via `python-dotenv`. Copy `.env.example` → `.env`.

Datasets follow **YOLO format** (`.yaml` config + `images/` + `labels/` structure). CVAT exports are typically in YOLO 1.1 or COCO JSON — check the format before parsing.

```python
from dotenv import load_dotenv
load_dotenv()
import os
data_root = os.environ["DATA_ROOT"]
```

## Device

Always abstract the device — never use raw `.cuda()`:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
tensor = tensor.to(device)
```

## Reproducibility

Use `cv_lib.train.set_seeds(seed)` or set manually:
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

```python
compare_gt_pred(image_path, model_path, class_names,
                conf_threshold=0.25, label_path=None,
                output_path=None, show=True) -> np.ndarray
```
Side-by-side GT (left) vs predictions (right). Auto-detects Jupyter vs terminal.
Label auto-resolve: same dir first, then `images/ → labels/` swap.

```python
load_yolo_gt(label_path, img_w, img_h) -> tuple[np.ndarray, np.ndarray]
# boxes_xyxy (N,4) float32, class_ids (N,) int32
```

Internal helpers (don't expose in public API): `_draw_boxes`, `_resolve_label_path`, `_add_panel_label`, `_in_notebook`.

### cv_lib.viz.batch

```python
show_batch(images, labels=None, class_names=None,
           tile_size=(320,320), cols=4,
           output_path=None, show=True) -> np.ndarray
```
- `images`: list of `np.ndarray` (H×W×C uint8 BGR), `torch.Tensor` (C×H×W float), or file paths
- `labels`: list of `(N,5)` arrays — `[class_id, cx, cy, w, h]` YOLO normalised
- Returns BGR grid; pads to full rows with black tiles

### cv_lib.viz.errors

```python
find_errors(images_dir, labels_dir=None,
            model_path=None, pred_labels_dir=None,
            conf_threshold=0.25, iou_threshold=0.5) -> list[ErrorEntry]

render_errors(entries, class_names=None,
              tile_size=(320,320), cols=4, max_tiles=32,
              output_path=None, show=True) -> np.ndarray
```
- Provide either `model_path` (live inference) or `pred_labels_dir` (pre-saved YOLO txts)
- `ErrorEntry`: `image_path`, `error_type` ("FP"/"FN"), `box_xyxy`, `class_id`, `conf`
- FP tiles drawn in red, FN in blue; cropped around the box with 30px padding

### cv_lib.data

```python
load_dataset_yaml(path) -> dict
class_names_from_yaml(path) -> list[str]
iter_image_label_pairs(images_dir, labels_dir=None, extensions=...) -> list[tuple[Path, Path]]
class_distribution(labels_dir, num_classes) -> np.ndarray  # shape (num_classes,)
data_root() -> Path  # reads DATA_ROOT env var, raises EnvironmentError if unset
```

### cv_lib.data.inspect

```python
inspect_dataset(images_dir, labels_dir=None, num_classes=None,
                class_names=None, extensions=...) -> InspectReport
```
- Checks: corrupt images (cv2.imread fails), missing label files, empty label files, invalid boxes
- Invalid box = class id OOB, center outside [0,1], size outside (0,1]
- Only fully valid boxes are counted in `class_counts`
- `report.print()` renders an ASCII class distribution bar chart

### cv_lib.data.convert

```python
cvat_xml_to_yolo(xml_path, out_dir, class_names=None) -> dict[str, int]
coco_json_to_yolo(json_path, out_dir, class_names=None) -> dict[str, int]
```
- Both return `{class_name: index}` mapping
- `class_names=None` → inferred from the file (CVAT labels / COCO categories)
- Output: one `.txt` per image in YOLO normalised format

### cv_lib.metrics

```python
plot_confusion_matrix(y_true, y_pred, class_names, normalize=True) -> Figure
summarize_map(results, class_names=None) -> dict[str, float]
# keys: "mAP50", "mAP50-95", "AP50/<name>"
```

`summarize_map` reads `results.box.map50`, `results.box.map`, `results.box.ap_class_index`, `results.box.ap` from Ultralytics `model.val()` return value.

### cv_lib.train

```python
set_seeds(seed=42) -> None
train(model_path, data, epochs=100, imgsz=640, batch=16, seed=42,
      project="runs/train", name="exp", device=None, **kwargs) -> Results
```
- `set_seeds` sets `random`, `numpy`, `torch`, `cudnn.deterministic`
- `train` saves `train_config.json` to `<project>/<name>/` before calling `model.train()`

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

## What's Next

- `viz/distribution.py` — class frequency bar chart (matplotlib Figure)
- `notebooks/` — usage examples for inspect, errors, batch viz
- CI workflow (GitHub Actions) — `uv run --extra dev pytest` on push

Don't add W&B/MLflow — Ultralytics writes W&B natively. Don't write a custom Dataset subclass until non-standard augmentation is required.
