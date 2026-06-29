# TODO — cv_lib

Roadmap для повседневного использования. Сгруппировано по приоритету.
Отметки: `[ ]` — не начато, `[~]` — в работе, `[x]` — готово.

> Предыдущий роадмап (P0–P3: CLI, CVAT CSV, DVC, split, distribution, augment,
> export, CI, тесты, pre-commit, notebooks) **полностью закрыт** — см. раздел
> [«Сделано»](#сделано-история) внизу. Ниже — новый бэклог под ежедневную работу
> с детекционными проектами.

---

## P0 — каждодневные операции, которых не хватает

- [x] **Tiled / sliced inference** `src/cv_lib/infer/tiled.py`
  `sliced_predict(model, image, tile=640, overlap=0.2, nms_iou=0.5, conf=0.25)` —
  режет крупный кадр на перекрывающиеся тайлы, инференсит, склеивает боксы обратно
  в координаты оригинала + глобальный NMS (SAHI-подход). Закрывает мелкие объекты на
  больших снимках. CLI: флаг `cvlib infer --tiled --tile 640 --overlap 0.2`.
  Тесты: `tests/test_infer_tiled.py` (склейка координат, дедуп на стыках).

- [x] **Ремап / фильтрация классов в YOLO-лейблах** `src/cv_lib/data/remap.py`
  `remap_labels(labels_dir, mapping, drop=None, out=None)` — слить/переименовать/выкинуть
  классы и пересчитать `data.yaml`. Частая операция при объединении датасетов или
  схлопывании таксономии. CLI: `cvlib remap labels/ --map 2=0 3=0 --drop 5`.
  Тесты: `tests/test_remap.py`, `tests/test_cli.py`.

- [x] **Annotation QA — поиск аномалий разметки** `src/cv_lib/data/qa.py`
  `audit_labels(labels_dir, ...) -> QAReport` — флагует подозрительные боксы (крошечные,
  во весь кадр, экстремальный aspect ratio), дубли боксов, выбросы по числу объектов на
  кадр, редкие классы. Дополняет `inspect` (тот ловит битое/OOB, этот — «странное, но
  валидное»). CLI: `cvlib qa labels/ --data data.yaml`. Тесты: `tests/test_qa.py`.

- [x] **Дедупликация и data-leakage** `src/cv_lib/data/dedup.py`
  `find_duplicates(images_dir, hamming=5) -> list[cluster]` на perceptual-hash (pHash);
  отдельно `check_split_leakage(split_dir)` — находит одинаковые/near-dup изображения
  между train/val/test (типичная причина «слишком хороших» метрик). CLI: `cvlib dedup`.
  Тесты: `tests/test_dedup.py`.

## P1 — оценка и выбор модели

- [x] **Per-class метрики + PR/F1-кривые** `src/cv_lib/metrics/__init__.py`
  `per_class_map(results) -> dict` и `plot_pr_curves(results, output_path=)` — разбивка
  mAP/precision/recall по классам + PR-кривые (где сейчас только confusion matrix и
  агрегатный `summarize_map`). Подключить к `cvlib eval --per-class`.
  Тесты: расширить `tests/test_metrics.py`.

- [x] **Подбор порога confidence / NMS** `src/cv_lib/metrics/threshold.py`
  `sweep_threshold(model, data, metric="f1") -> ThresholdReport` — перебор `conf` (и
  опц. `iou`), F1-vs-confidence, рекомендованная рабочая точка. Чтобы не ставить порог
  на глаз перед деплоем. CLI: `cvlib threshold --model best.pt --data data.yaml`.
  Тесты: `tests/test_threshold.py`.

- [x] **Кросс-форматный бенч экспортов** расширить `src/cv_lib/cli/_bench.py`
  PyTorch vs ONNX vs TensorRT в одной таблице: latency/FPS **и** mAP-паритет
  (через `validate_export`), чтобы видеть цену ускорения в качестве. CLI:
  `cvlib bench --model best.pt --formats pt onnx trt`. Тесты: `tests/test_bench.py` (моки).

## P1 — данные и лейблинг

- [x] **Авто-предразметка (pre-annotation)** `src/cv_lib/data/autolabel.py`
  `autolabel(model, images_dir, out, conf=0.4) -> int` — прогон модели по неразмеченным
  кадрам → YOLO `.txt` для импорта в CVAT как черновик. Ускоряет ручной лейблинг в разы.
  CLI: `cvlib autolabel --model best.pt images/ --out labels/`. Тесты: `tests/test_autolabel.py`.

- [x] **Hard-example mining / приоритет аннотации** `src/cv_lib/data/mining.py`
  `rank_for_labeling(model, images_dir, by="uncertainty") -> list[(path, score)]` —
  ранжирует неразмеченный пул по неуверенности/низкому conf/числу детекций, чтобы
  размечать сначала полезное (active learning). CLI: `cvlib mine`. Тесты: `tests/test_mining.py`.

- [x] **Извлечение кропов объектов** `src/cv_lib/data/crops.py`
  `extract_crops(images_dir, labels_dir, out, per_class=True, pad=0.0)` — вырезает объекты
  по боксам (раскладка `out/<class>/...`) для ревью качества разметки или датасета под
  классификатор. CLI: `cvlib crops images/ labels/ --out crops/`. Тесты: `tests/test_crops.py`.

## P2 — медиа и отчётность

- [ ] **Видео-инференс** `src/cv_lib/infer/video.py`
  `predict_video(model, src, out=None, save_labels=False, stride=1)` — поток кадров →
  аннотированное видео и/или per-frame YOLO-лейблы. CLI: `cvlib infer clip.mp4 --out out.mp4`.
  Тесты: `tests/test_infer_video.py` (синтетический клип из нескольких кадров).

- [ ] **Сводный отчёт по eval** `src/cv_lib/report.py`
  `build_report(eval_results, output="report.html")` — единый HTML/Markdown: mAP-таблица +
  confusion matrix + per-class + FP/FN тайлы (`viz.errors`) + class distribution. Чтобы
  делиться результатом прогона одним файлом. CLI: `cvlib report`. Тесты: `tests/test_report.py`.

- [x] **Экспорт YOLO → COCO / Pascal VOC** расширить `src/cv_lib/data/convert.py`
  `yolo_to_coco(...)` / `yolo_to_voc(...)` — обратное направление к существующему импорту,
  для интеропа и сабмишенов. CLI: `cvlib convert labels/ --to coco --out ann.json`.
  Тесты: расширить `tests/test_data_convert.py` (round-trip YOLO↔COCO).

---

## Сделано (история)

**P0 — issue #1/#2/#3:** единый CLI `cvlib`, сборка как библиотеки (lazy public API),
CVAT CSV → YOLO + `cvat-query`, генерация `dvc.yaml`.
**P1:** train/val/test split + `data.yaml`, class distribution chart, превью аугментаций
(`augment_preview`), `cvlib export` (ONNX/TensorRT).
**P2:** CI workflow (ruff + pytest), покрытие тестами core-модулей, pre-commit, notebook-примеры.
**P3:** `__version__` + `cvlib --version`, структурное логирование (loguru/`--verbose`),
сравнение прогонов (`compare-runs`).
**Daily-ops:** tiled inference, remap/qa/dedup/crops, per-class метрики + PR-кривые,
threshold sweep, кросс-форматный бенч, autolabel, mining, YOLO→COCO/VOC.
**Асимметрии:** `cvlib train` (CLI к `train()`), VOC→YOLO импорт, `cvlib merge`
(слияние датасетов), `cvlib fix` (`repair_labels` — клип/дроп невалидных боксов).

---

### Не делаем (зафиксировано в CLAUDE.md)
- ❌ W&B / MLflow — Ultralytics пишет W&B нативно.
- ❌ Кастомный `Dataset`-сабкласс — пока нет нестандартной аугментации.
