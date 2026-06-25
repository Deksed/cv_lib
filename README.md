# cv-lib

CV utility library for iterating on object detection models.

**Stack:** Ultralytics (YOLO) · PyTorch · OpenCV · Albumentations

---

## Requirements

| | |
|---|---|
| Python | 3.10 – 3.12 |
| CUDA | ≥ 11.8 (целевая машина) |
| GPU | опционально; CPU-fallback встроен |

---

## Setup

### uv (рекомендуется)

```bash
# 1. Установить uv (если ещё нет)
pip install uv

# 2. Установить torch с cu118-индекса + всё остальное
uv pip install -e ".[cu118,dev]"
```

`[tool.uv.sources]` в `pyproject.toml` автоматически берёт `torch` и `torchvision`
с `https://download.pytorch.org/whl/cu118` — дополнительных флагов не нужно.

Для разработки на машине с CUDA 12.4+ (Python 3.13) — замените `cu118` на прямую установку:

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
uv pip install -e ".[dev]"
```

### pip

```bash
pip install -r requirements-torch.txt
pip install -e ".[dev]"
```

---

## Переменные окружения

Скопируйте `.env.example` в `.env` и заполните:

```bash
cp .env.example .env
```

| Переменная | Назначение |
|---|---|
| `DATA_ROOT` | корень датасета; относительные пути к изображениям резолвятся от него |
| `CVAT_ANNOTATIONS_PATH` | путь к CVAT-экспорту (YOLO 1.1 / COCO JSON) |

Все скрипты подгружают `.env` автоматически при запуске.

---

## Структура

```
cv_lib/
├── src/cv_lib/
│   ├── cli/                # cvlib CLI: inspect/convert/compare/infer/eval/bench/compare-runs
│   ├── viz/
│   │   ├── compare.py      # GT vs prediction side-by-side
│   │   ├── batch.py        # show_batch() — грид изображений с боксами
│   │   └── errors.py       # find_errors() / render_errors() — FP/FN тайлы
│   ├── data/
│   │   ├── __init__.py     # YOLO-формат, class distribution, iter pairs
│   │   ├── inspect.py      # проверка датасета: битые, пропущенные, OOB
│   │   └── convert.py      # CVAT XML / COCO JSON → YOLO txt
│   ├── metrics/
│   │   └── __init__.py     # confusion matrix, mAP summary
│   ├── train/
│   │   └── __init__.py     # train() — сиды + config snapshot + model.train()
│   └── export.py           # ONNX / TensorRT export + validate
├── scripts/
│   ├── eval.py             # model.val() → mAP-таблица + confusion matrix PNG
│   ├── batch_infer.py      # батч-инференс → YOLO-лейблы и/или изображения
│   └── compare_gt_pred.py  # CLI-обёртка над viz.compare
├── tests/                  # 24 теста, pytest
├── notebooks/
├── configs/
├── .env.example
├── pyproject.toml
└── requirements-torch.txt
```

---

## CLI

После установки пакета доступна единая команда `cvlib` (entry point из `pyproject.toml`):

```bash
cvlib --help
cvlib <command> --help
```

| Команда | Назначение |
|---|---|
| `cvlib inspect` | health-check датасета (битые/пропущенные/невалидные боксы) |
| `cvlib convert` | CVAT XML / COCO JSON / CVAT CSV → YOLO `.txt` |
| `cvlib cvat-query` | поиск/фильтрация по CVAT CSV (label/task/assignee/image) |
| `cvlib compare` | GT vs prediction side-by-side для одного изображения |
| `cvlib infer`   | батч-инференс → YOLO-лейблы и/или аннотированные изображения |
| `cvlib eval`    | `model.val()` → mAP-таблица + confusion matrix |
| `cvlib bench`   | sanity-check + бенчмарк латентности/FPS |
| `cvlib compare-runs` | сравнение train-прогонов: конфиги + лучшие метрики |

У всех подкоманд есть флаг `--verbose` (DEBUG-логи через loguru); статус-сообщения
идут в stderr, форматированные таблицы — в stdout. Версия: `cvlib --version` / `cvlib -V`.

```bash
cvlib inspect dataset/images/val --data dataset/data.yaml
cvlib convert annotations.xml --out labels/ --names car person
cvlib convert cvat_export.csv --out labels/          # CVAT CSV → YOLO
cvlib cvat-query cvat_export.csv --label car --assignee anna --count
cvlib eval --model runs/train/best.pt --data dataset/data.yaml
cvlib bench --model best.pt --imgsz 320 640 1280
```

Скрипты в `scripts/` — тонкие обёртки над теми же командами (общий код в `cv_lib.cli`),
оставлены для обратной совместимости. `cvlib eval ...` ≡ `python scripts/eval.py ...`.

---

## Scripts

### `scripts/eval.py`

Запускает `model.val()`, печатает mAP-таблицу, сохраняет confusion matrix.

```bash
python scripts/eval.py --model runs/train/best.pt --data dataset/data.yaml

# Кастомный путь для confusion matrix:
python scripts/eval.py --model best.pt --data data.yaml --cm-out eval/cm.png

# Только метрики, без confusion matrix:
python scripts/eval.py --model best.pt --data data.yaml --no-cm

# Оценить на тестовом сплите:
python scripts/eval.py --model best.pt --data data.yaml --split test
```

### `scripts/batch_infer.py`

Прогоняет модель по директории изображений.

```bash
# Сохранить YOLO-лейблы:
python scripts/batch_infer.py --model best.pt --images dataset/images/val --save-labels

# Сохранить аннотированные изображения:
python scripts/batch_infer.py --model best.pt --images dataset/images/val --save-vis

# Оба варианта:
python scripts/batch_infer.py --model best.pt --images dataset/images/val \
    --data dataset/data.yaml --save-labels --save-vis --out-dir runs/infer
```

### `scripts/compare_gt_pred.py`

Side-by-side сравнение GT и предсказаний для одного изображения.

```bash
python scripts/compare_gt_pred.py frame.jpg \
    --model runs/train/best.pt \
    --names car person bike

# Прочитать классы из data.yaml:
python scripts/compare_gt_pred.py frame.jpg --model best.pt --data dataset/data.yaml

# Сохранить без показа (headless):
python scripts/compare_gt_pred.py frame.jpg --model best.pt --names car \
    --output compare.jpg --no-show
```

---

## API

### `cv_lib.viz`

```python
from cv_lib.viz import compare_gt_pred, load_yolo_gt, show_batch, find_errors, render_errors
```

#### `compare_gt_pred()`

```python
result = compare_gt_pred(
    image_path="data/images/frame_042.jpg",
    model_path="runs/train/best.pt",
    class_names=["car", "person"],
    conf_threshold=0.25,
    output_path="out.jpg",
    show=True,
)  # → np.ndarray BGR
```

#### `show_batch()`

```python
from cv_lib.viz.batch import show_batch

grid = show_batch(
    images=[img1, img2, img3],          # ndarray, Tensor или путь к файлу
    labels=[lbl1, lbl2, lbl3],          # (N,5) — [class_id, cx, cy, w, h] YOLO
    class_names=["car", "person"],
    tile_size=(320, 320),
    cols=4,
    output_path="grid.png",
    show=False,
)  # → np.ndarray BGR
```

#### `find_errors()` / `render_errors()`

```python
from cv_lib.viz.errors import find_errors, render_errors

errors = find_errors(
    images_dir="dataset/images/val",
    pred_labels_dir="runs/infer/labels",  # или model_path= для live-инференса
    conf_threshold=0.25,
    iou_threshold=0.5,
)
grid = render_errors(errors, class_names=["car"], output_path="errors.png", show=False)
```

---

### `cv_lib.data`

```python
from cv_lib.data import (
    load_dataset_yaml,        # dict из data.yaml
    class_names_from_yaml,    # list[str]
    iter_image_label_pairs,   # list[(image_path, label_path)]
    class_distribution,       # np.ndarray (N_classes,)
    data_root,                # Path из DATA_ROOT env var
)
```

#### `inspect_dataset()`

```python
from cv_lib.data.inspect import inspect_dataset

report = inspect_dataset(
    images_dir="dataset/images/val",
    num_classes=3,
    class_names=["car", "person", "bike"],
)
report.print()
# → итог: total images, corrupt, missing labels, invalid boxes, class distribution
```

#### `cvat_xml_to_yolo()` / `coco_json_to_yolo()`

```python
from cv_lib.data.convert import cvat_xml_to_yolo, coco_json_to_yolo

cvat_xml_to_yolo("annotations.xml", out_dir="labels/", class_names=["car", "person"])
coco_json_to_yolo("instances_val.json", out_dir="labels/")
```

---

### `cv_lib.metrics`

```python
from cv_lib.metrics import plot_confusion_matrix, summarize_map

fig = plot_confusion_matrix(y_true, y_pred, class_names=["car", "person"])

results = model.val(data="data.yaml")
summary = summarize_map(results)
# → {"mAP50": 0.83, "mAP50-95": 0.61, "AP50/car": 0.91, ...}
```

---

### `cv_lib.train`

```python
from cv_lib.train import train

results = train(
    model_path="yolov8n.pt",
    data="dataset/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    seed=42,
    project="runs/train",
    name="exp1",
)
# Сохраняет train_config.json в runs/train/exp1/ перед стартом
```

---

### `cv_lib.export`

```python
from cv_lib.export import export_onnx, export_trt, validate_export

onnx_path = export_onnx(model, "weights/best.onnx")
engine_path = export_trt("weights/best.onnx", "weights/best.engine", fp16=True)
validate_export(pytorch_out, onnx_out, atol=1e-4)
```

---

## Code Quality

```bash
ruff check src/ scripts/
ruff check --fix src/ scripts/
ruff format src/ scripts/
```

Правила: `E` · `F` · `I` · `UP`. Длина строки — 100 символов.

---

## Tests

```bash
uv run --extra dev pytest
uv run --extra dev pytest --cov=cv_lib --cov-report=term-missing
```

24 теста, покрывают `data.inspect`, `data.convert`, `viz.batch`, `viz.errors`.
