# TODO — cv_lib

Roadmap для повседневного использования. Сгруппировано по приоритету.
Отметки: `[ ]` — не начато, `[~]` — в работе, `[x]` — готово.

Связано с открытыми issue: **#1** (сборка как библиотека), **#2** (CVAT CSV → YOLO),
**#3** (генерация `dvc.yaml`).

---

## P0 — закрывает открытые issue, разблокирует ежедневную работу

- [x] **Единый CLI `cvlib`** (#1)
  `[project.scripts]` → `cv_lib.cli:main`; пакет `src/cv_lib/cli/` с подкомандами
  `inspect`, `convert`, `compare`, `infer`, `eval`, `bench`. Скрипты в `scripts/` стали
  тонкими шимами над общим кодом (обратная совместимость, без дублирования). Тесты: `tests/test_cli.py`.

- [x] **Инструкция по сборке и установке как библиотеки** (#1)
  Раздел в README «Сборка и использование как библиотеки»: `uv build` / `pip wheel`,
  установка wheel в стороннем проекте, пример `from cv_lib import ...`. Публичный API
  зафиксирован в `cv_lib/__init__.py` (ленивый реэкспорт через `__getattr__`/`__all__`).
  Тесты: `tests/test_public_api.py`.

- [x] **Поддержка CVAT CSV формата** (#2)
  `cvat_csv_to_yolo()` + `query_cvat_csv()` в `data/convert.py` (+ `CVAT_CSV_COLUMNS`).
  CLI: `cvlib convert *.csv` (автодетект формата) и `cvlib cvat-query` (фильтр по
  label/task/assignee/image). Фильтр по `instance_shape` (по умолчанию rectangles).
  Тесты: `tests/test_data_convert.py`, `tests/test_cli.py`.

- [ ] **Генерация `dvc.yaml`** (#3)
  `src/cv_lib/data/dvc_gen.py` + `cvlib dvc-init`. Стандартные стейджи:
  `collect → split → train (внешний модуль) → compare-with-prev → report`.

## P1 — частые операции, которых сейчас нет

- [ ] **Сплит датасета** `src/cv_lib/data/split.py`
  `train_val_test_split(images_dir, labels_dir, ratios=(0.8,0.1,0.1), seed=42, stratify_by_class=True)`
  + генерация `data.yaml`. Нужен и сам по себе, и как стейдж `split` для DVC (#3).

- [ ] **Превью аугментаций** `src/cv_lib/viz/augment.py`
  `albumentations` уже в зависимостях, но нигде не используется. Грид «оригинал vs N аугментаций»
  с корректным пересчётом боксов — чтобы быстро проверять пайплайн перед обучением.

- [ ] **Class distribution chart** `src/cv_lib/viz/distribution.py`
  Из «What's Next» в CLAUDE.md. matplotlib `Figure` с частотой классов (train/val рядом).

- [ ] **`scripts/export.py` + `cvlib export`**
  CLI-обёртка над `cv_lib.export` (ONNX/TensorRT + `validate_export`). Сейчас export только программно.

## P2 — качество и воспроизводимость

- [ ] **CI workflow** `.github/workflows/ci.yml`
  Из «What's Next»: `ruff check` + `uv run --extra dev pytest` на push/PR (CPU-only матрица).

- [ ] **Добить покрытие тестами**
  Тесты есть для `data.inspect`, `data.convert`, `viz.batch`, `viz.errors`.
  Нет для: `viz.compare`, `metrics`, `train` (snapshot конфига), `export` (с моком), `data/__init__`.

- [ ] **pre-commit hook**
  `.pre-commit-config.yaml` с `ruff check --fix` + `ruff format`, чтобы не гонять руками.

- [ ] **Notebook-примеры** `notebooks/`
  Из «What's Next»: рабочие примеры `inspect_dataset`, `find_errors/render_errors`, `show_batch`.
  Сейчас только `viz_test.ipynb`.

## P3 — приятные мелочи

- [x] **`__version__`** в `cv_lib/__init__.py` + `cvlib --version` / `-V` (PR #4 + интеграция в CLI).
- [x] **Структурное логирование** — loguru, общий `--verbose` (`add_verbose`/`setup_logging`),
  статус-сообщения через `logger`, таблицы остаются на `print` (PR #4 + перенос в пакет `cli/`).
- [x] **Сравнение прогонов** — `scripts/compare_runs.py` + подкоманда `cvlib compare-runs`
  (configs из `train_config.json` + лучшие метрики из `results.csv`).

---

### Не делаем (зафиксировано в CLAUDE.md)
- ❌ W&B / MLflow — Ultralytics пишет W&B нативно.
- ❌ Кастомный `Dataset`-сабкласс — пока нет нестандартной аугментации.
