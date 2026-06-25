"""`cvlib dvc-init` — scaffold a DVC pipeline (dvc.yaml + params.yaml)."""

from __future__ import annotations

import argparse

from loguru import logger

from cv_lib.cli._common import add_verbose

HELP = "Scaffold a DVC pipeline (dvc.yaml + params.yaml) for the train/eval loop."

EPILOG = (
    "Stages: collect -> split -> train -> compare-with-prev -> report.\n"
    "The result is a starting point — edit paths/commands to match your dataset.\n"
    "cv_lib does not depend on DVC; install it separately (`pip install dvc`).\n\n"
    "Examples:\n"
    "  cvlib dvc-init\n"
    "  cvlib dvc-init --annotations data/raw/ann.json --run-name yolov8n_640\n"
    "  cvlib dvc-init --stages collect split train --no-params --force\n"
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    from cv_lib.data.dvc_gen import STAGES

    parser.add_argument("--out", default="dvc.yaml", metavar="PATH", help="dvc.yaml path.")
    parser.add_argument(
        "--params-out", default="params.yaml", metavar="PATH", help="params.yaml path."
    )
    parser.add_argument(
        "--no-params", action="store_true", help="Do not write params.yaml."
    )
    parser.add_argument(
        "--stages", nargs="+", choices=STAGES, default=list(STAGES), metavar="STAGE",
        help=f"Stages to include (default: all — {' '.join(STAGES)}).",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing files.")

    # Path/command overrides (all optional; sensible defaults otherwise).
    parser.add_argument("--annotations", metavar="PATH", help="Raw annotations (collect input).")
    parser.add_argument("--images", metavar="DIR", help="Images directory.")
    parser.add_argument("--labels", metavar="DIR", help="YOLO labels directory (collect output).")
    parser.add_argument("--dataset", metavar="DIR", help="Split dataset directory.")
    parser.add_argument("--data", metavar="YAML", help="Path to data.yaml.")
    parser.add_argument("--runs-root", metavar="DIR", help="Training runs root.")
    parser.add_argument("--run-name", metavar="NAME", help="Training run name.")
    parser.add_argument("--train-cmd", metavar="CMD", help="External training command.")
    parser.add_argument("--cm-out", metavar="PATH", help="Confusion matrix output (report).")
    add_verbose(parser)


def run(args: argparse.Namespace) -> None:
    from cv_lib.data.dvc_gen import PipelineConfig, generate_dvc_yaml, generate_params_yaml

    # Only override config fields the user actually passed.
    overrides = {
        "annotations": args.annotations,
        "images_dir": args.images,
        "labels_dir": args.labels,
        "dataset_dir": args.dataset,
        "data_yaml": args.data,
        "runs_root": args.runs_root,
        "run_name": args.run_name,
        "train_cmd": args.train_cmd,
        "cm_out": args.cm_out,
    }
    config = PipelineConfig(**{k: v for k, v in overrides.items() if v is not None})

    try:
        dvc_path = generate_dvc_yaml(
            args.out, config=config, stages=tuple(args.stages), force=args.force
        )
        logger.info("Wrote {} ({} stages)", dvc_path, len(args.stages))
        print(f"Wrote {dvc_path}")

        if not args.no_params and "train" in args.stages:
            params_path = generate_params_yaml(args.params_out, force=args.force)
            logger.info("Wrote {}", params_path)
            print(f"Wrote {params_path}")
    except FileExistsError as exc:
        raise SystemExit(str(exc)) from exc

    print("Next: review the file(s), then `pip install dvc && dvc repro`.")
