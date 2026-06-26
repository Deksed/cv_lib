# CLAUDE.md

Guidance for Claude Code working in this repo. For API signatures see README.md
and the module docstrings — don't duplicate them here.

## Project

CV utility library for iterating on object detection models. Stack: **Ultralytics
(YOLO)** primary, PyTorch + torchvision for custom bits, OpenCV for image I/O and
visualization, Jupyter for exploration.

```
src/cv_lib/
├── __init__.py              # public API: lazy re-exports (PEP 562 __getattr__) + __all__
├── cli/                     # cvlib unified CLI (entry point `cvlib`)
│   ├── __init__.py          # build_parser() + main(); COMMANDS registry
│   ├── __main__.py          # `python -m cv_lib.cli`
│   ├── _common.py           # configure_console, load_env, setup_logging, add_verbose, resolve_names
│   └── _<cmd>.py            # one per subcommand: HELP + add_arguments(p) + run(args), opt. EPILOG
├── viz/                     # compare.py, batch.py, errors.py, distribution.py (GT/pred, grids, FP/FN tiles, class-freq chart)
├── data/                    # __init__ (YOLO parsing), inspect.py, convert.py, split.py, dvc_gen.py
├── metrics/__init__.py      # confusion matrix, mAP summary
├── train/__init__.py        # train() wrapper: seeds + config snapshot + model.train()
└── export.py                # ONNX / TensorRT export + validate
scripts/                     # thin shims over cv_lib.cli (backwards compat; prefer `cvlib`)
tests/                       # pytest; conftest.py has sample_image / yolo_label_file fixtures
configs/                     # YOLO data.yaml templates (currently empty)
notebooks/                   # Jupyter experiments
.env.example                 # env var template — copy to .env
```

## Public API

`cv_lib/__init__.py` re-exports the public surface lazily via `__getattr__`
(PEP 562): `import cv_lib` stays cheap (no eager OpenCV/torch/Ultralytics) and
`from cv_lib import inspect_dataset, ...` works. `__all__` is the contract —
anything not listed is internal. **When adding a public function, register it in
`_EXPORTS`** (and the `TYPE_CHECKING` block for IDE hints).

## CLI

Single entry point `cvlib` (`[project.scripts]` → `cv_lib.cli:main`):
`cvlib inspect|convert|cvat-query|split|distribution|compare|infer|eval|export|bench|compare-runs|dvc-init`.

Add a subcommand: create `cv_lib/cli/_<name>.py` with `HELP` / `add_arguments(parser)`
/ `run(args)` (opt. `EPILOG`) and register it in `COMMANDS` in `cli/__init__.py`.
Scripts in `scripts/` are thin shims — never duplicate logic there.

- `configure_console()` forces UTF-8 output (box-drawing `─`/`█` break under cp1251).
- Logging: `add_verbose(parser)` adds `--verbose`; `setup_logging(verbose)` configures
  loguru (DEBUG/INFO → stderr). Status messages via `logger`; formatted tables stay on
  `print` (stdout). `main()` and shims call `setup_logging` after parsing.

## Setup

Target machine: Python 3.12, CUDA ≥ 11.8 → cu118 torch (2.4–2.5).

```bash
uv pip install -e ".[cu118,dev]"        # uv (recommended); [tool.uv.sources] pulls torch from cu118
# pip alternative:
pip install -r requirements-torch.txt && pip install -e ".[dev]"
```

CUDA 12.4+ / Python 3.13 → install cu124 torch manually and drop `<3.13` from
`requires-python`. Lint: `ruff check src/ scripts/ tests/`. Tests: `uv run --extra dev pytest`.
Build wheel: `uv build`. Local gate: `uv run pre-commit install` (runs ruff on commit).

## Data & Annotations

Dataset root and annotation paths are **never hardcoded** — read from `DATA_ROOT`
and `CVAT_ANNOTATIONS_PATH` env vars (or config params). `.env` is loaded
automatically (`python-dotenv`); copy `.env.example` → `.env`.

Datasets follow **YOLO format** (`.yaml` + `images/` + `labels/`). CVAT exports are
YOLO 1.1 XML, COCO JSON, or flat CVAT CSV — check the format before parsing.

## Device

Always abstract the device — never raw `.cuda()`:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device); tensor = tensor.to(device)
```

## Reproducibility

Use `cv_lib.set_seeds(seed)` (sets `random`, `numpy`, `torch`, `cudnn.deterministic`).
`cv_lib.train()` writes `train_config.json` to `<project>/<name>/` before `model.train()`.

## Conventions

**Ultralytics**
- Pass the `.yaml` to `model.train(data=...)`, not raw paths.
- Use the Results API (`results.boxes`, `results.masks`) over raw tensor indexing.

**Visualization** (`src/cv_lib/viz/`)
- Draw on a **copy** (`img.copy()`) — never mutate the input.
- Return `np.ndarray` (BGR, uint8) — OpenCV-compatible, not tensors.
- Accept both `np.ndarray` (H×W×C) and `torch.Tensor` (C×H×W float); normalize internally.

**Export** (`src/cv_lib/export.py`)
- Use `export_onnx` / `export_trt` / `validate_export` — don't call the Ultralytics
  export API directly in scripts. Always `validate_export(..., atol=1e-4)` after export.
- `tensorrt` is not a project dependency — install separately.

**Code style**
- Type hints on all public functions.
- Ruff rules `E`, `F`, `I`, `UP`; line length 100; `E501` ignored. Notebooks exempt from `E402`/`F401`.

## What's Next

- `notebooks/` — usage examples (inspect, errors, batch viz)
- CI workflow (GitHub Actions) — `ruff check` + `uv run --extra dev pytest` on push

Don't add W&B/MLflow — Ultralytics writes W&B natively. Don't write a custom
`Dataset` subclass until non-standard augmentation is required.
