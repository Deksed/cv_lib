# cv-lib

[![CI](https://github.com/Deksed/cv_lib/actions/workflows/ci.yml/badge.svg)](https://github.com/Deksed/cv_lib/actions/workflows/ci.yml)

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

### Закрытый контур / офлайн (air-gapped)

`requirements-lock.txt` — полностью запиненный lock со всеми транзитивными
зависимостями (core + cu118-torch) и хешами. Универсальный: содержит маркеры
платформ (`sys_platform`, `platform_machine`), поэтому годится и для Linux-целевой
машины, и для Windows. Сам пакет `cv-lib` в lock не входит — он собирается локально.

На машине **с** доступом в интернет скачать все колёса:

```bash
pip download -r requirements-lock.txt -d wheelhouse/   # + сам wheel: uv build / pip wheel . --no-deps -w wheelhouse/
```

Перенести `wheelhouse/` в закрытый контур и поставить **без сети**:

```bash
pip install --no-index --find-links wheelhouse/ -r requirements-lock.txt
pip install --no-index --find-links wheelhouse/ cv_lib-0.1.0-py3-none-any.whl
```

> Для сборки wheel в контуре нужны ещё build-зависимости из `[build-system]`
> (`setuptools>=68`, `wheel`) — добавьте их в `wheelhouse/` тем же `pip download`.
> Пересобрать lock: `uv pip compile pyproject.toml --extra cu118 --universal
> --python-version 3.12 --generate-hashes --no-emit-package cv-lib -o requirements-lock.txt`.

---

## Сборка и использование как библиотеки

`cv_lib` можно собрать в wheel/sdist и поставить в сторонний проект.

### Сборка дистрибутива

```bash
uv build                 # → dist/cv_lib-<версия>-py3-none-any.whl + .tar.gz
# или, без uv:
pip wheel . --no-deps -w dist/
```

`uv build` кладёт в `dist/` оба артефакта: wheel (`*.whl`) и sdist (`*.tar.gz`).

### Установка в стороннем проекте

```bash
pip install cv_lib-0.1.0-py3-none-any.whl
```

> `torch`/`torchvision` не входят в зависимости wheel (ставятся отдельно с нужного
> CUDA-индекса — см. Setup). Установите их в целевом окружении до или после wheel.

### Публичный API

Всё, что нужно в коде, реэкспортится с верхнего уровня пакета — импортируйте прямо из `cv_lib`:

```python
import cv_lib
cv_lib.__version__                       # "0.1.0"

from cv_lib import (
    # viz
    compare_gt_pred, load_yolo_gt, show_batch, find_errors, render_errors, ErrorEntry,
    plot_class_distribution, augment_preview, default_transform,
    # data
    load_dataset_yaml, class_names_from_yaml, iter_image_label_pairs,
    class_distribution, data_root,
    inspect_dataset, InspectReport,
    cvat_xml_to_yolo, coco_json_to_yolo, cvat_csv_to_yolo, query_cvat_csv,
    cvat_csv_gt, predictions_to_cvat_csv,
    # data — train/val/test split
    train_val_test_split, SplitReport,
    # data — DVC pipeline scaffolding
    build_pipeline, generate_dvc_yaml, generate_params_yaml, PipelineConfig,
    # metrics
    plot_confusion_matrix, summarize_map,
    # train
    set_seeds, train,
    # export
    export_onnx, export_trt, validate_export,
)
```

Имена импортируются **лениво** (PEP 562): `import cv_lib` дёшев и не тянет
OpenCV/torch/Ultralytics, пока конкретная функция не понадобится. Полный список —
`cv_lib.__all__`; всё, чего там нет, считается внутренним и может меняться без
предупреждения. Подмодули (`cv_lib.viz`, `cv_lib.data`, `cv_lib.metrics`,
`cv_lib.train`, `cv_lib.export`, `cv_lib.cli`) тоже импортируются напрямую.

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
│   ├── cli/                # cvlib CLI: inspect/fix/convert/split/merge/distribution/augment/remap/qa/dedup/crops/cvat-query/compare/infer/autolabel/mine/train/eval/threshold/export/bench/compare-runs/dvc-init
│   ├── viz/
│   │   ├── compare.py      # GT vs prediction side-by-side
│   │   ├── batch.py        # show_batch() — грид изображений с боксами
│   │   ├── errors.py       # find_errors() / render_errors() — FP/FN тайлы
│   │   ├── distribution.py # plot_class_distribution() — бар-чарт частоты классов
│   │   └── augment.py      # augment_preview() — original-vs-aug грид (боксы пересчитываются)
│   ├── data/
│   │   ├── __init__.py     # YOLO-формат, class distribution, iter pairs
│   │   ├── inspect.py      # проверка датасета: битые, пропущенные, OOB
│   │   ├── convert.py      # CVAT/COCO/CSV/VOC → YOLO txt и YOLO → COCO/VOC
│   │   ├── split.py        # train/val/test split + data.yaml (стратификация)
│   │   ├── merge.py        # merge_datasets() — слить датасеты с выравниванием классов
│   │   ├── repair.py       # repair_labels() — авто-починка боксов (клип/дроп)
│   │   ├── remap.py        # remap_labels() — слить/переименовать/выкинуть классы
│   │   ├── qa.py           # audit_labels() — аномалии разметки
│   │   ├── dedup.py        # near-duplicate (pHash) + data-leakage между сплитами
│   │   ├── crops.py        # extract_crops() — кропы объектов по классам
│   │   ├── autolabel.py    # autolabel() — авто-предразметка моделью
│   │   ├── mining.py       # rank_for_labeling() — hard-example mining
│   │   └── dvc_gen.py      # генерация dvc.yaml / params.yaml (DVC-пайплайн)
│   ├── infer/
│   │   └── tiled.py        # sliced_predict() — tiled inference для крупных кадров
│   ├── metrics/
│   │   ├── __init__.py     # confusion matrix, mAP, per_class_map, PR-кривые
│   │   └── threshold.py    # sweep_threshold() — подбор рабочего порога
│   ├── train/
│   │   └── __init__.py     # train() — сиды + config snapshot + model.train()
│   └── export.py           # ONNX / TensorRT export + validate
├── scripts/
│   ├── eval.py             # model.val() → mAP-таблица + confusion matrix PNG
│   ├── batch_infer.py      # батч-инференс → YOLO-лейблы и/или изображения
│   └── compare_gt_pred.py  # CLI-обёртка над viz.compare
├── tests/                  # 165 тестов, pytest
├── notebooks/
├── configs/
├── .env.example
├── pyproject.toml
├── requirements-torch.txt   # torch/torchvision с cu118-индекса (pip)
└── requirements-lock.txt    # полный pinned lock + хеши для офлайн/закрытого контура
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
| `cvlib fix` | авто-починка лейблов: клип боксов за границы, дроп невалидных |
| `cvlib convert` | CVAT XML / COCO JSON / CVAT CSV / VOC ↔ YOLO `.txt` (+ YOLO → COCO/VOC через `--to`) |
| `cvlib split` | train/val/test split YOLO-датасета + генерация `data.yaml` |
| `cvlib merge` | слить несколько YOLO-датасетов в один (выравнивание классов по имени) |
| `cvlib distribution` | бар-чарт частоты классов (сравнение train/val/test) |
| `cvlib augment` | превью аугментаций: original vs N вариантов (боксы пересчитываются) |
| `cvlib remap` | слить/переименовать/выкинуть классы в YOLO-лейблах (+ rewrite `data.yaml`) |
| `cvlib qa` | аудит разметки: крошечные/огромные боксы, экстремальный aspect, дубли, выбросы |
| `cvlib dedup` | near-duplicate изображения (pHash) и data-leakage между train/val/test |
| `cvlib crops` | вырезать объекты по боксам в `out/<class>/` (ревью / датасет под классификатор) |
| `cvlib cvat-query` | поиск/фильтрация по CVAT CSV (label/task/assignee/image) |
| `cvlib compare` | GT vs prediction side-by-side для одного изображения |
| `cvlib infer`   | батч-инференс → YOLO-лейблы, аннотированные изображения и/или CVAT CSV (`--cvat-csv`, `--template`; `--tiled` для крупных кадров) |
| `cvlib autolabel` | авто-предразметка модели → YOLO `.txt` черновики для CVAT |
| `cvlib mine` | ранжирование неразмеченных по неуверенности (active-learning очередь) |
| `cvlib train`   | обучение YOLO с сидами + снапшотом `train_config.json` |
| `cvlib eval`    | `model.val()` → mAP-таблица + confusion matrix (`--per-class` для разбивки) |
| `cvlib threshold` | sweep confidence → рекомендованная рабочая точка (F1/P/R) |
| `cvlib export`  | экспорт YOLO `.pt` → ONNX (или сборка TensorRT `.engine`) |
| `cvlib bench`   | sanity-check + бенчмарк латентности/FPS (`--formats pt onnx trt`) |
| `cvlib compare-runs` | сравнение train-прогонов: конфиги + лучшие метрики |
| `cvlib dvc-init` | генерация `dvc.yaml` (+ `params.yaml`) — DVC-пайплайн train/eval |

У всех подкоманд есть флаг `--verbose` (DEBUG-логи через loguru); статус-сообщения
идут в stderr, форматированные таблицы — в stdout. Версия: `cvlib --version` / `cvlib -V`.

```bash
cvlib inspect dataset/images/val --data dataset/data.yaml
cvlib convert annotations.xml --out labels/ --names car person
cvlib convert cvat_export.csv --out labels/          # CVAT CSV → YOLO
cvlib cvat-query cvat_export.csv --label car --assignee anna --count
cvlib convert voc_xml/ --format voc --out labels/ --names car person   # Pascal VOC → YOLO
cvlib fix dataset/labels/train --num-classes 5 --out fixed/        # клип/дроп невалидных боксов
cvlib split dataset/images --out dataset_split --names car person   # train/val/test + data.yaml
cvlib merge dsA dsB dsC --out merged                              # слить датасеты (классы по имени)
cvlib distribution dataset_split --out class_dist.png              # частота классов по сплитам
cvlib augment img.jpg --labels img.txt --names car person --out aug.png   # превью аугментаций
cvlib remap labels/ --map 2=0 3=0 --drop 5 --out remapped/        # слить/выкинуть классы
cvlib train --model yolov8n.pt --data dataset/data.yaml --epochs 100    # обучение + снапшот конфига
cvlib qa dataset_split/labels/train                               # аудит подозрительной разметки
cvlib dedup dataset_split --leakage                               # утечки между сплитами
cvlib crops dataset/images --labels dataset/labels --out crops/   # кропы объектов по классам
cvlib autolabel images/ --model best.pt --out labels/            # черновая разметка для CVAT
cvlib mine images/ --model best.pt --top 20                       # что размечать в первую очередь
cvlib eval --model runs/train/best.pt --data dataset/data.yaml --per-class
cvlib threshold --model best.pt --data dataset/data.yaml          # подбор рабочего порога
cvlib export runs/train/best.pt --format onnx --imgsz 640        # → best.onnx
cvlib bench --model best.pt --formats pt onnx --imgsz 640         # кросс-форматный бенч
cvlib convert dataset/labels --to coco --images dataset/images --names car person --out ann.json
cvlib dvc-init                                       # → dvc.yaml + params.yaml
```

### DVC-пайплайн

`cvlib dvc-init` генерирует **скаффолд** `dvc.yaml` со стандартными стейджами
`collect → split → train → compare-with-prev → report` и `params.yaml` с
гиперпараметрами обучения. cv_lib не зависит от DVC — поставьте его отдельно
(`pip install dvc`), отредактируйте пути под свой датасет и запустите `dvc repro`.

```bash
cvlib dvc-init --run-name yolov8n_640                # переопределить имя прогона
cvlib dvc-init --stages collect split train --no-params --force
```

Программно — через `cv_lib.dvc_gen`:

```python
from cv_lib import PipelineConfig, generate_dvc_yaml, generate_params_yaml

cfg = PipelineConfig(annotations="data/raw/ann.json", run_name="exp1")
generate_dvc_yaml("dvc.yaml", config=cfg)
generate_params_yaml("params.yaml")
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
    model_path="runs/train/best.pt",   # путь .pt ИЛИ уже загруженный YOLO (переиспользование)
    class_names=["car", "person"],     # опц.: если None — берётся из predict (results.names)
    conf_threshold=0.25,
    axis="vertical",                   # "horizontal" (по умолч.) | "vertical" для широких кадров
    output_path="out.jpg",
    show=True,
)  # → np.ndarray BGR; печатает путь/ссылку CVAT и уверенность по каждому боксу
```

Чтобы не перезагружать веса на каждый кадр, загрузи модель один раз и передавай
объект:

```python
from ultralytics import YOLO
model = YOLO("runs/train/best.pt")
for frame in frames:
    compare_gt_pred(frame, model, show=False, output_path=f"cmp/{frame.name}")
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

#### `plot_class_distribution()`

```python
from cv_lib.viz.distribution import plot_class_distribution

# Один набор лейблов:
fig = plot_class_distribution("dataset/labels/train", class_names=["car", "person"])

# Сравнение сплитов (грид баров рядом), сортировка по частоте, сохранение:
fig = plot_class_distribution(
    {"train": "ds/labels/train", "val": "ds/labels/val", "test": "ds/labels/test"},
    class_names=["car", "person"],
    sort=True,            # по убыванию общей частоты
    horizontal=True,      # горизонтальные бары (удобно для длинных имён)
    output_path="class_dist.png",
)  # → matplotlib Figure
```

`num_classes` и имена выводятся из лейблов, если не заданы. Возвращает `Figure`
(в ноутбуке отображается сам, в скрипте — `fig.savefig(...)`).

#### `augment_preview()` / `default_transform()`

Превью `albumentations`-пайплайна на одном изображении: `original` плюс `N`
аугментированных вариантов, тайлами в грид. Боксы трансформируются вместе с
пикселями (`BboxParams(format="yolo")`) и пересчитываются на каждом тайле — удобно
убедиться, что геометрические преобразования не «теряют» аннотации, **до** того как
пайплайн уйдёт в обучение.

```python
import numpy as np
from cv_lib.viz.augment import augment_preview, default_transform

boxes = np.array([[0, 0.5, 0.5, 0.4, 0.3]])   # [class_id, cx, cy, w, h] (YOLO-norm)

# Дефолтный пайплайн (flip + photometric + affine):
grid = augment_preview(
    "frame.jpg", boxes,
    class_names=["car", "person"],
    n=8,                       # число вариантов (original показывается первым)
    seed=42,                   # вариант i использует seed+i → детерминированно
    output_path="aug.png",
)  # → np.ndarray (BGR, uint8)

# Свой пайплайн (при боксах обязан нести bbox_params):
import albumentations as A
my = A.Compose(
    [A.RandomScale(p=1.0), A.RandomBrightnessContrast(p=1.0)],
    bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
)
grid = augment_preview("frame.jpg", boxes, transform=my, show=False)
```

`image` принимает путь, BGR `np.ndarray` (H×W×C) или float `torch.Tensor` (C×H×W).
Без `boxes_yolo` работает на неразмеченном изображении. `default_transform()` уже
содержит `bbox_params` (боксы, видимые <20% после трансформа, отбрасываются).
Детерминизм обеспечивается через `transform.set_random_seed(seed+i)` (albumentations
≥1.4.21), с фолбэком на глобальные RNG для старых версий.

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

#### `predictions_to_cvat_csv()` — обратный путь: предсказания → CVAT CSV

Прогоняет Ultralytics-модель по папке и пишет детекции в плоский CVAT CSV
(`CVAT_CSV_COLUMNS`), готовый к заливке обратно в CVAT для ручной правки.
По умолчанию rectangle; для seg-модели `masks="auto"` пишет `polygon`-строки с
`instance_points` (`"x1,y1;x2,y2;…"`), `bbox_*` при этом тоже заполнен.
Служебные поля, которых нет у модели (`image_id`, `job_id`, `task_id`,
`task_name`, `task_assignee`, `image_path` + любые доп. колонки вроде ссылки на
CVAT), джойнятся из шаблонного CSV по `image_name`:

```python
from cv_lib.data.convert import predictions_to_cvat_csv

n_rows = predictions_to_cvat_csv(
    "best.pt",                       # путь к .pt или загруженный YOLO
    images_dir="dataset/images/val",
    out_csv="runs/preann/export.csv",
    template_csv="cvat_template.csv",  # опц.: метаданные/ссылки по image_name
    class_names=["car", "person"],   # иначе берётся из model.names
    conf=0.25, imgsz=640,
    masks="auto",                    # seg-модель → polygon-строки; False = только bbox
    save_conf=True,                  # доп. колонка "confidence" для фильтрации
)
# → export.csv: по строке на детекцию; кадры без детекций строк не дают
```

То же из CLI (переиспользует уже посчитанные детекции, без второго прохода):

```bash
cvlib infer --model best.pt --images dataset/images/val \
    --cvat-csv runs/preann/export.csv --template cvat_template.csv --cvat-conf
```

Для готовой YOLO-разметки (а не предсказаний) — симметричный экспорт
`yolo_to_cvat_csv()` (как `yolo_to_coco`/`yolo_to_voc`):

```python
from cv_lib.data.convert import yolo_to_cvat_csv

yolo_to_cvat_csv(
    images_dir="dataset/images/val",
    labels_dir="dataset/labels/val",   # inferred from images path if None
    out_csv="gt.csv",
    class_names=["car", "person"],
    template_csv="cvat_template.csv",  # опц.
)
# CLI: cvlib convert labels/ --to cvat-csv --images images/ --names car person --out gt.csv
```

#### `cvat_csv_gt()` — GT для отрисовки прямо из CVAT CSV

CVAT CSV как источник истины для путей и боксов (туда удобно класть ссылку на
CVAT). Возвращает словарь `image_name → {image_path, width, height, boxes (xyxy),
class_ids, labels, meta}`; `meta` — первая строка кадра со всеми колонками
(включая ссылку на CVAT):

```python
from cv_lib.data.convert import cvat_csv_gt

records = cvat_csv_gt("export.csv", class_names=["car", "person"])
rec = records["frame_001.jpg"]
print(rec["image_path"], rec["boxes"], rec["labels"])
print(rec["meta"].get("cvat_url"))   # доп. колонка из CSV, если есть

# compare_gt_pred берёт путь и GT из CSV, а не из YOLO-датасета:
from cv_lib.viz import compare_gt_pred
compare_gt_pred(
    "frame_001.jpg",                 # имя; путь возьмётся из CSV, если файла нет рядом
    model_path="best.pt",
    class_names=["car", "person"],
    csv_path="export.csv",           # GT + путь + печать ссылки на CVAT из строки
)
```

#### `train_val_test_split()`

```python
from cv_lib.data.split import train_val_test_split

report = train_val_test_split(
    images_dir="dataset/images",
    labels_dir="dataset/labels",   # inferred from images path if omitted
    out_dir="dataset_split",
    ratios=(0.8, 0.1, 0.1),        # 2 values → train/val only
    seed=42,
    stratify_by_class=True,        # bucket by dominant class, keeps rare classes
    class_names=["car", "person"], # else inferred as ["0", "1", ...]
    mode="copy",                   # "copy" | "symlink" | "move"
)
report.print()
# → <out>/images/{train,val,test}, <out>/labels/{train,val,test}, <out>/data.yaml
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

82 теста, покрывают `data.inspect`, `data.convert`, `data.split`, `data.dvc_gen`,
`viz.batch`, `viz.errors`, `viz.distribution`, `viz.augment`, публичный API
(`cv_lib.__all__`) и CLI (`cvlib`).
